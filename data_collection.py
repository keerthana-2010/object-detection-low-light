import carla
import random
import queue
import numpy as np
import cv2
from pascal_voc_writer import Writer
import os
import time

#Connecting to the CARLA Server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.load_world('Town10HD') # can be changed to town5
bp_lib = world.get_blueprint_library()

original_settings = world.get_settings()

#spawn ego vehicle
print("Spawning ego vehicle...")
ego_bp = random.choice(bp_lib.filter('vehicle'))
spawn_points = world.get_map().get_spawn_points()
ego_spawn_point = random.choice(spawn_points)
ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)

if ego_vehicle is None:
    print("Failed to spawn ego vehicle!")
    exit(1)

ego_vehicle.set_autopilot(True)

# Enable synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

# Spawn RGB Camera
rgb_bp = bp_lib.find('sensor.camera.rgb')
rgb_bp.set_attribute('image_size_x', '800')
rgb_bp.set_attribute('image_size_y', '600')
rgb_bp.set_attribute('fov', '90')
camera_init_trans = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=-15))
rgb_camera = world.spawn_actor(rgb_bp, camera_init_trans, attach_to=ego_vehicle)

# Spawn Semantic Segmentation Camera
seg_bp = bp_lib.find('sensor.camera.semantic_segmentation')
seg_bp.set_attribute('image_size_x', '800')
seg_bp.set_attribute('image_size_y', '600')
seg_bp.set_attribute('fov', '90')
seg_camera = world.spawn_actor(seg_bp, camera_init_trans, attach_to=ego_vehicle)

if rgb_camera is None or seg_camera is None:
    print("Camera spawn failed.")
    ego_vehicle.destroy()
    exit(1)

# Now that camera is spawned, get its transform and forward vector
camera_transform = rgb_camera.get_transform()
camera_location = camera_transform.location
camera_forward = camera_transform.get_forward_vector()
#spawning walkers
num_pedestrians = 150 # number of walkers
walkers = []
walker_controllers = []
walker_bp_list = bp_lib.filter('walker.pedestrian.*')

ego_transform = ego_spawn_point  # carla.Transform
ego_location = ego_transform.location
ego_forward_vector = ego_transform.get_forward_vector()
right_vector = ego_transform.get_right_vector()

min_spacing = 0.3  # ~30 cm between pedestrians, almost touching

start_forward = 20.0  # start 20 meters ahead of ego
start_lateral = 0.0   # all pedestrians in one line (no lateral offset)

ped_spawn_points = []

for i in range(num_pedestrians):
    forward_offset = start_forward + i * min_spacing
    lateral_offset = start_lateral

    spawn_loc = ego_location + ego_forward_vector * forward_offset + right_vector * lateral_offset
    spawn_loc.z += 1.0  # raise slightly so they get spawned above the ground

    ped_spawn_points.append(carla.Transform(spawn_loc))

# Spawn walkers
for spawn_point in ped_spawn_points:
    walker_bp = random.choice(walker_bp_list)
    walker = world.try_spawn_actor(walker_bp, spawn_point)
    if walker:
        walkers.append(walker)

# Spawn controllers with targets nearby
walker_controller_bp = bp_lib.find('controller.ai.walker')
map = world.get_map()
all_waypoints = map.generate_waypoints(1.0)

target_radius = 20
pedestrian_targets = [wp.transform.location for wp in all_waypoints if wp.transform.location.distance(ego_location) <= target_radius]
if not pedestrian_targets:
    pedestrian_targets = [wp.transform.location for wp in all_waypoints]

walker_controllers = []
for walker in walkers:
    controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
    walker_controllers.append(controller)
    controller.start()
    controller.go_to_location(random.choice(pedestrian_targets))
    controller.set_max_speed(1.4 + random.random())

print(f"Spawned {len(walkers)} pedestrians and {len(walker_controllers)} controllers.")


# Spawn Vehicles
num_vehicles = 20  # number of vehicles

vehicle_spawn_points = world.get_map().get_spawn_points()
vehicle_spawn_points = [pt for pt in vehicle_spawn_points if pt.location.distance(ego_spawn_point.location) > 10]  # Spawn vehicles at least 10m away from ego vehicle

vehicles = []
vehicle_bp_list = bp_lib.filter('vehicle.*')

vehicle_spawned = 0
attempts = 0
max_attempts = num_vehicles * 10

while vehicle_spawned < num_vehicles and attempts < max_attempts:
    attempts += 1
    spawn_point = random.choice(vehicle_spawn_points)
    # Avoid too close vehicles 
    too_close = False
    for v in vehicles:
        if v.get_transform().location.distance(spawn_point.location) < 5.0:  # minimum 5m spacing
            too_close = True
            break
    if too_close:
        continue

    vehicle_bp = random.choice(vehicle_bp_list)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        vehicle.set_autopilot(True)  # make vehicles drive autonomously
        vehicles.append(vehicle)
        vehicle_spawned += 1


# Image queues
rgb_queue = queue.Queue()
seg_queue = queue.Queue()

rgb_camera.listen(lambda image: rgb_queue.put_nowait(image))
seg_camera.listen(lambda image: seg_queue.put_nowait(image))


world.tick()

BASE_DIR = "output"
os.makedirs(f'{BASE_DIR}/images', exist_ok=True)
os.makedirs(f'{BASE_DIR}/labels', exist_ok=True)
os.makedirs(f'{BASE_DIR}/annotated', exist_ok=True)

"""3d to 2d projections 
https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/ """

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = np.array([point_camera[1], -point_camera[2], point_camera[0]])
    if point_camera[2] <= 0:
        return np.array([-1, -1])
    point_img = np.dot(K, point_camera)
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]
    return point_img[0:2]

def point_in_canvas(pos, img_h, img_w):
    return 0 <= pos[0] < img_w and 0 <= pos[1] < img_h

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def non_max_suppression(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        boxes = [b for b in boxes if iou(current, b) < iou_threshold]
    return keep

image_w = int(rgb_bp.get_attribute("image_size_x").as_int())
image_h = int(rgb_bp.get_attribute("image_size_y").as_int())
fov = float(rgb_bp.get_attribute("fov").as_float())
K = build_projection_matrix(image_w, image_h, fov)

#number of frames to be collected
max_frames = 1000
frame_count = 0
class_counts = {
    "vehicle": 0,
    "walker": 0,
    "tree": 0
}


try:
    while frame_count < max_frames:
        print(f"Tick {frame_count}")
        world.tick()

        try:
            rgb_img = rgb_queue.get(timeout=10)
            seg_img = seg_queue.get(timeout=10)
        except queue.Empty:
            print("Timeout waiting for images.")
            break

        # Convert raw BGRA data to BGR OpenCV image
        img_bgra = np.reshape(np.copy(rgb_img.raw_data), (image_h, image_w, 4))
        img_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        img_seg = np.reshape(np.copy(seg_img.raw_data), (image_h, image_w, 4))

        # Save RGB image
        image_filename = f'demo/images/{frame_count:06d}.png'
        cv2.imwrite(image_filename, img_rgb)

        writer = Writer(image_filename, image_w, image_h)

        world_2_camera = np.array(rgb_camera.get_transform().get_inverse_matrix())
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()

        all_boxes = []

        """semantic segmentation id
        https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera """

        # Detect walkers
        for walker in walkers:
            if not walker.is_alive:
                continue

            bb = walker.bounding_box
            walker_location = walker.get_transform().location
            dist = walker_location.distance(ego_location)
            if dist > 50:
                continue
            ray = walker_location - ego_location
            if ego_forward.dot(ray) <= 0:
                continue

            verts = bb.get_world_vertices(walker.get_transform())
            proj_verts = [get_image_point(v, K, world_2_camera) for v in verts]
            xs = [p[0] for p in proj_verts if point_in_canvas(p, image_h, image_w)]
            ys = [p[1] for p in proj_verts if point_in_canvas(p, image_h, image_w)]

            if not xs or not ys:
                continue

            x_min = max(0, int(min(xs)))
            y_min = max(0, int(min(ys)))
            x_max = min(image_w, int(max(xs)))
            y_max = min(image_h, int(max(ys)))

            if x_max > x_min and y_max > y_min:
                crop = img_seg[y_min:y_max, x_min:x_max]
                red_channel = crop[:, :, 2]
                pedestrian_pixels = np.sum(red_channel == 12)  # class 12 = walker
                total_pixels = crop.shape[0] * crop.shape[1]
                if total_pixels == 0 or pedestrian_pixels / total_pixels < 0.1:
                    continue


                all_boxes.append([x_min, y_min, x_max, y_max, 1])

        # Detect vehicles
        for vehicle in vehicles:
            if not vehicle.is_alive:
                continue

            bb = vehicle.bounding_box
            vehicle_location = vehicle.get_transform().location
            dist = vehicle_location.distance(ego_location)
            if dist > 70:  # increased distance threshold
                continue

            ray = vehicle_location - ego_location
    # Relax forward check or remove to include parked cars on sides
    # if ego_forward.dot(ray) <= 0:
    #     continue

            verts = bb.get_world_vertices(vehicle.get_transform())
            proj_verts = [get_image_point(v, K, world_2_camera) for v in verts]
            xs = [p[0] for p in proj_verts if point_in_canvas(p, image_h, image_w)]
            ys = [p[1] for p in proj_verts if point_in_canvas(p, image_h, image_w)]

            if not xs or not ys:
                continue

            x_min = max(0, int(min(xs)))
            y_min = max(0, int(min(ys)))
            x_max = min(image_w, int(max(xs)))
            y_max = min(image_h, int(max(ys)))

            if x_max > x_min and y_max > y_min:
                crop = img_seg[y_min:y_max, x_min:x_max]
                red_channel = crop[:, :, 2]
                vehicle_pixels = np.sum(np.isin(red_channel, [13,14,15,16,17,18,19])) 
                total_pixels = crop.shape[0] * crop.shape[1]
                pixel_ratio = vehicle_pixels / total_pixels if total_pixels > 0 else 0
                if total_pixels == 0 or (pixel_ratio < 0.03 and total_pixels < 500):  # relaxed thresholds
                    continue

                all_boxes.append([x_min, y_min, x_max, y_max, 0])




        # Detect trees from semantic segmentation
        tree_mask = (img_seg[:, :, 2] == 9).astype(np.uint8)
        tree_mask = cv2.morphologyEx(tree_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(tree_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue

            all_boxes.append([x, y, x+w, y+h, 2])

        # Apply NMS class-wise
        final_boxes = []
        for class_id in [0,1,2]:  # vehicle=0, walker=1, tree=2
            class_boxes = [box for box in all_boxes if box[4] == class_id]
            nms_boxes = non_max_suppression(class_boxes, iou_threshold=0.4)
            final_boxes.extend(nms_boxes)

        # Write to Pascal VOC format
        class_names = {0: "vehicle", 1: "walker", 2: "tree"}
        for box in final_boxes:
            x_min, y_min, x_max, y_max, class_id = box
            class_name = class_names[class_id]
            writer.addObject(class_name, x_min, y_min, x_max, y_max)
            class_counts[class_name] += 1

    # Draw bounding box
            color = (0, 255, 0) if class_name == "walker" else (255, 0, 0) if class_name == "vehicle" else (0, 128, 255)
            cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(img_rgb, class_name, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        writer.save(f'demo/labels/{frame_count:06d}.xml')
        cv2.imwrite(f'demo/annotated/{frame_count:06d}.png', img_rgb)


        frame_count += 1

finally:
    print("Cleaning up actors...")
    rgb_camera.stop()
    seg_camera.stop()

    for controller in walker_controllers:
        controller.stop()
        controller.destroy()

    for walker in walkers:
        walker.destroy()

    for vehicle in vehicles:
        vehicle.destroy()

    if ego_vehicle is not None:
        ego_vehicle.destroy()

    # Restore original settings
    world.apply_settings(original_settings)

    print("Cleanup done.")

    print("\n=== Total Class Counts ===") # to count the class instances
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}") 


