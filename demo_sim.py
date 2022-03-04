import os
import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)
from mot import MultiObjectTracker
from alfred.utils.log import logger
from alfred.dl.torch.common import device
from nosmpl.box_trans import convert_vertices_to_ori_img, get_box_scale_info
from realrender.render import render_human_mesh
import glob


MIN_NUM_FRAMES = 25


def main(args):
    video_file = args.vid_file
    output_path = os.path.join(
        args.output_folder, os.path.basename(video_file).replace(".mp4", "")
    )
    output_track_res_f = os.path.join(output_path, "track_result.pkl")
    images_folder = os.path.join(
        args.output_folder, os.path.basename(video_file).replace(".mp4", ""), "raw"
    )
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    images_list = glob.glob(os.path.join(images_folder, "*.png"))
    if len(images_list) > 0:
        image_folder = images_folder
        num_frames = len(images_list)
        img_shape = cv2.imread(images_list[0]).shape[:-1]
        logger.info("Found existed extracted images.")
    else:
        image_folder, num_frames, img_shape = video_to_images(
            video_file, img_folder=images_folder, return_info=True
        )

    print(f"Input video number of frames {num_frames}")
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()
    logger.info("start detection and tracking...")
    bbox_scale = 1.1
    
    if os.path.exists(output_track_res_f):
        tracking_results = joblib.load(output_track_res_f)
    else:
        if args.tracking_method == "pose":
            if not os.path.isabs(video_file):
                video_file = os.path.join(os.getcwd(), video_file)
            tracking_results = run_posetracker(
                video_file, staf_folder=args.staf_dir, display=args.display
            )
        else:
            # run multi object tracker
            mot = MultiObjectTracker()
            tracking_results = mot.track(image_folder, show=False)
            joblib.dump(tracking_results, output_track_res_f)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]["frames"].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file, map_location="cpu")
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt["gen_state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from "{pretrained_file}"')

    print(f"Running VIBE on each tracklet...")
    vibe_time = time.time()
    vibe_results = {}

    person_ids = list(tracking_results.keys())
    i = 0
    for person_id in person_ids:
        bboxes = joints2d = None
        if args.tracking_method == "bbox":
            bboxes = tracking_results[person_id]["bbox"]
        elif args.tracking_method == "pose":
            joints2d = tracking_results[person_id]["joints2d"]

        frames = tracking_results[person_id]["frames"]

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=args.vibe_batch_size, num_workers=10)

        with torch.no_grad():
            (
                pred_cam,
                pred_verts,
                pred_pose,
                pred_betas,
                pred_joints3d,
                smpl_joints2d,
                norm_joints2d,
            ) = ([], [], [], [], [], [], [])

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(
                    output["theta"][:, :, :3].reshape(batch_size * seqlen, -1)
                )
                pred_verts.append(output["verts"].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(
                    output["theta"][:, :, 3:75].reshape(batch_size * seqlen, -1)
                )
                pred_betas.append(
                    output["theta"][:, :, 75:].reshape(batch_size * seqlen, -1)
                )
                pred_joints3d.append(
                    output["kp_3d"].reshape(batch_size * seqlen, -1, 3)
                )
                smpl_joints2d.append(
                    output["kp_2d"].reshape(batch_size * seqlen, -1, 2)
                )

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
            del batch

        # ========= [Optional] run Temporal SMPLify to refine the results ========= #
        if args.run_smplify and args.tracking_method == "pose":
            norm_joints2d = np.concatenate(norm_joints2d, axis=0)
            norm_joints2d = convert_kps(norm_joints2d, src="staf", dst="spin")
            norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

            # Run Temporal SMPLify
            (
                update,
                new_opt_vertices,
                new_opt_cam,
                new_opt_pose,
                new_opt_betas,
                new_opt_joints3d,
                new_opt_joint_loss,
                opt_joint_loss,
            ) = smplify_runner(
                pred_rotmat=pred_pose,
                pred_betas=pred_betas,
                pred_cam=pred_cam,
                j2d=norm_joints2d,
                device=device,
                batch_size=norm_joints2d.shape[0],
                pose2aa=False,
            )

            # update the parameters after refinement
            print(
                f"Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}"
            )
            pred_verts = pred_verts.cpu()
            pred_cam = pred_cam.cpu()
            pred_pose = pred_pose.cpu()
            pred_betas = pred_betas.cpu()
            pred_joints3d = pred_joints3d.cpu()
            pred_verts[update] = new_opt_vertices[update]
            pred_cam[update] = new_opt_cam[update]
            pred_pose[update] = new_opt_pose[update]
            pred_betas[update] = new_opt_betas[update]
            pred_joints3d[update] = new_opt_joints3d[update]

        elif args.run_smplify and args.tracking_method == "bbox":
            print(
                "[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!"
            )
            print("[WARNING] Continuing without running Temporal SMPLify!..")

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        smpl_joints2d = smpl_joints2d.cpu().numpy()

        # Runs 1 Euro Filter to smooth out the results
        if args.smooth:
            min_cutoff = args.smooth_min_cutoff  # 0.004
            beta = args.smooth_beta  # 1.5
            print(
                f"Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}"
            )
            pred_verts, pred_pose, pred_joints3d = smooth_pose(
                pred_pose, pred_betas, min_cutoff=min_cutoff, beta=beta
            )

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam, bbox=bboxes, img_width=orig_width, img_height=orig_height
        )

        joints2d_img_coord = convert_crop_coords_to_orig_img(
            bbox=bboxes,
            keypoints=smpl_joints2d,
            crop_size=224,
        )
        i += 1
        print(f"\r{i}/{len(person_ids)}", end="", flush=True)

        output_dict = {
            "pred_cam": pred_cam,
            "orig_cam": orig_cam,
            "verts": pred_verts,
            "pose": pred_pose,
            "betas": pred_betas,
            "joints3d": pred_joints3d,
            "joints2d": joints2d,
            "joints2d_img_coord": joints2d_img_coord,
            "bboxes": bboxes,
            "frame_ids": frames,
        }
        vibe_results[person_id] = output_dict

    del model

    end = time.time()
    fps = num_frames / (end - vibe_time)

    print(f"VIBE FPS: {fps:.2f}")
    total_time = time.time() - total_time
    print(f"Total time spent: {total_time:.2f} seconds (including model loading time).")
    print(f"Total FPS (including model loading time): {num_frames / total_time:.2f}.")

    print(f'Saving output results to "{os.path.join(output_path, "vibe_output.pkl")}".')
    joblib.dump(vibe_results, os.path.join(output_path, "vibe_output.pkl"))

    if not args.no_render:
        renderer = Renderer(
            resolution=(orig_width, orig_height),
            orig_img=True,
            wireframe=args.wireframe,
        )

        output_img_folder = f"{image_folder}_output"
        os.makedirs(output_img_folder, exist_ok=True)

        print(f"Rendering output video, writing frames to {output_img_folder}")

        # prepare results for rendering
        frame_results = prepare_rendering_results(vibe_results, num_frames)
        mesh_color = {
            k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
            for k in vibe_results.keys()
        }

        image_file_names = sorted(
            [
                os.path.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith(".png") or x.endswith(".jpg")
            ]
        )

        for frame_idx in tqdm(range(len(image_file_names))):
            img_fname = image_file_names[frame_idx]
            img = cv2.imread(img_fname)

            if args.sideview:
                side_img = np.zeros_like(img)

            for person_id, person_data in frame_results[frame_idx].items():
                frame_verts = person_data["verts"]
                frame_cam = person_data["cam"]
                pred_cam = person_data["pred_cam"]
                bboxes = person_data["bboxes"]

                box_scale_o2n, box_topleft, _ = get_box_scale_info(img, bboxes)

                mc = mesh_color[person_id]
                mesh_filename = None
                save_obj = True
                if save_obj:
                    mesh_folder = os.path.join("output", "meshes", f"{person_id:04d}")
                    os.makedirs(mesh_folder, exist_ok=True)
                    mesh_filename = os.path.join(mesh_folder, f"{frame_idx:06d}.obj")

                frame_verts = convert_vertices_to_ori_img(
                    frame_verts, pred_cam[0], pred_cam[1:], box_scale_o2n, box_topleft
                )
                tri = renderer.faces.astype(np.int32)
                frame_verts[:, 2] = -frame_verts[:, 2]
                img = render_human_mesh(
                    img, [frame_verts], tri, alpha=0.9, color=mc, with_bg_flag=True
                )

            cv2.imwrite(os.path.join(output_img_folder, f"{frame_idx:06d}.png"), img)
            cv2.imshow("Video", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if args.display:
            cv2.destroyAllWindows()

        # ========= Save rendered video ========= #
        vid_name = os.path.basename(video_file)
        save_name = f'{vid_name.replace(".mp4", "")}_vibe_result.mp4'
        save_name = os.path.join(output_path, save_name)
        print(f"Saving result video to {save_name}")
        images_to_video(img_folder=output_img_folder, output_vid_file=save_name)
        shutil.rmtree(output_img_folder)

    shutil.rmtree(image_folder)
    print("================= END =================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vid_file", type=str, help="input video path or youtube link")

    parser.add_argument(
        "--output_folder", type=str, help="output folder to write results"
    )

    parser.add_argument(
        "--tracking_method",
        type=str,
        default="bbox",
        choices=["bbox", "pose"],
        help="tracking method to calculate the tracklet of a subject from the input video",
    )

    parser.add_argument(
        "--detector",
        type=str,
        default="yolo",
        choices=["yolo", "maskrcnn"],
        help="object detector to be used for bbox tracking",
    )

    parser.add_argument(
        "--yolo_img_size",
        type=int,
        default=416,
        help="input image size for yolo detector",
    )

    parser.add_argument(
        "--tracker_batch_size",
        type=int,
        default=12,
        help="batch size of object detector used for bbox tracking",
    )

    parser.add_argument(
        "--staf_dir",
        type=str,
        default="/home/mkocabas/developments/openposetrack",
        help="path to directory STAF pose tracking method installed.",
    )

    parser.add_argument(
        "--vibe_batch_size", type=int, default=450, help="batch size of VIBE"
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="visualize the results of each step during demo",
    )

    parser.add_argument(
        "--run_smplify",
        action="store_true",
        help="run smplify for refining the results, you need pose tracking to enable it",
    )

    parser.add_argument(
        "--no_render",
        action="store_true",
        help="disable final rendering of output video.",
    )

    parser.add_argument(
        "--wireframe", action="store_true", help="render all meshes as wireframes."
    )

    parser.add_argument(
        "--sideview",
        action="store_true",
        help="render meshes from alternate viewpoint.",
    )

    parser.add_argument(
        "--save_obj", action="store_true", help="save results as .obj files."
    )

    parser.add_argument(
        "--smooth", action="store_true", help="smooth the results to prevent jitter"
    )

    parser.add_argument(
        "--smooth_min_cutoff",
        type=float,
        default=0.004,
        help="one euro filter min cutoff. "
        "Decreasing the minimum cutoff frequency decreases slow speed jitter",
    )

    parser.add_argument(
        "--smooth_beta",
        type=float,
        default=0.7,
        help="one euro filter beta. "
        "Increasing the speed coefficient(beta) decreases speed lag.",
    )

    args = parser.parse_args()

    main(args)
