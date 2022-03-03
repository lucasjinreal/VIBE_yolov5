from operator import ipow
from realrender.render import render_human_mesh
import numpy as np
import cv2
import os
import tqdm
import colorsys
from lib.utils.demo_utils import prepare_rendering_results
import pickle
import joblib
from lib.utils.renderer import Renderer
import glob
import trimesh
from nosmpl.box_trans import convert_vertices_to_ori_img, get_box_scale_info


vibe_results = joblib.load(open("output/blackpink/vibe_output.pkl", "rb"))
output_img_folder = f"output/render_out/1"
image_folder = "output/blackpink_mp4"


img_files = glob.glob(os.path.join(image_folder, "*.png"))
print("img files: ", len(img_files))

a = cv2.imread(img_files[0])
orig_height, orig_width, _ = a.shape

renderer = Renderer(
    resolution=(orig_width, orig_height), orig_img=True, wireframe=False
)

os.makedirs(output_img_folder, exist_ok=True)

print(f"Rendering output video, writing frames to {output_img_folder}")

# prepare results for rendering
# print(vibe_results[372])
frame_results = prepare_rendering_results(vibe_results, 4320)
mesh_color = {
    k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()
}

image_file_names = sorted(
    [
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith(".png") or x.endswith(".jpg")
    ]
)

for frame_idx in range(len(image_file_names)):
    img_fname = image_file_names[frame_idx]
    img = cv2.imread(img_fname)

    # if args.sideview:
    #     side_img = np.zeros_like(img)

    for person_id, person_data in frame_results[frame_idx].items():
        frame_verts = person_data["verts"]
        frame_cam = person_data["cam"]
        pred_cam = person_data["pred_cam"]
        bboxes = person_data["bboxes"]
        print("bboxes: ", bboxes)
        # pred_cam = convert_crop_cam_to_orig_img(frame_cam, bboxes, orig_width, orig_height)
        print(pred_cam)
        
        box_scale_o2n, box_topleft, _ = get_box_scale_info(img, bboxes)

        mc = mesh_color[person_id]
        mesh_filename = None
        save_obj = True
        if save_obj:
            mesh_folder = os.path.join("output", "meshes", f"{person_id:04d}")
            os.makedirs(mesh_folder, exist_ok=True)
            mesh_filename = os.path.join(mesh_folder, f"{frame_idx:06d}.obj")

        print(frame_verts, frame_verts.dtype, frame_verts.shape)
        frame_verts = convert_vertices_to_ori_img(
            frame_verts, pred_cam[0], pred_cam[1:], box_scale_o2n, box_topleft
        )
        print(mc)
        tri = renderer.faces.astype(np.int32)
        frame_verts[:, 2] = -frame_verts[:, 2]
        img = render_human_mesh(img, [frame_verts], tri, alpha=0.9, color=mc, with_bg_flag=True)

    cv2.imwrite(os.path.join(output_img_folder, f"{frame_idx:06d}.png"), img)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
