import carla
import cv2
import numpy as np
import torch
import random
import os
import time

# Class labels (Pascal VOC / detection classes)
CLASS_NAMES = {
    0: "vehicle",
    1: "pedestrian",
    2: "tree"
}

# BGR colors for visualization (OpenCV format)
CLASS_COLORS = {
    0: (255, 0, 0),   # vehicle → blue
    1: (0, 255, 0),   # pedestrian → green
    2: (0, 128, 255)  # tree → orange
}

frame = None  # global camera frame

# Persistent detection store
persistent_boxes = {}  # {id: {'box': [x1, y1, x2, y2], 'class': cls_id, 'frames_left': int}}
BOX_LIFETIME = 5       # number of frames to keep undetected boxes

def process_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

def camera_callback(image):
    global frame
    frame = process_image(image)

def set_night_weather(world):
    weather = carla.WeatherParameters(
        sun_altitude_angle=-15.0,
        cloudiness=20.0,
        precipitation=0.0,
        fog_density=0.3,
        fog_distance=30.0,
        wetness=0.2
    )
    world.set_weather(weather)
    print(" Night environment set.")

def spawn_pedestrians(world, blueprint_library, client, count=30):
    walker_bp = blueprint_library.filter('walker.pedestrian.*')
    controller_bp = blueprint_library.find('controller.ai.walker')
    spawn_points = []

    for _ in range(count * 2):
        loc = world.get_random_location_from_navigation()
        if loc:
            spawn_points.append(carla.Transform(loc))
        if len(spawn_points) >= count:
            break

    if not spawn_points:
        print(" No pedestrian spawn points.")
        return [], []

    batch = [carla.command.SpawnActor(random.choice(walker_bp), sp) for sp in spawn_points[:count]]
    results = client.apply_batch(batch)

    if results is None:
        print(" apply_batch returned None.")
        return [], []

    walkers, controllers = [], []
    for r in results:
        if r.error:
            continue
        walker = world.get_actor(r.actor_id)
        if not walker:
            continue
        controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        if controller:
            controller.start()
            dest = world.get_random_location_from_navigation()
            if dest:
                controller.go_to_location(dest)
            controller.set_max_speed(1 + random.random())
            walkers.append(walker)
            controllers.append(controller)

    return walkers, controllers

def spawn_vehicles(world, blueprint_library, count=15):
    vehicle_bps = [bp for bp in blueprint_library.filter('vehicle.*') if 'bike' not in bp.id.lower()]
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    vehicles = []

    for sp in spawn_points[:count]:
        bp = random.choice(vehicle_bps)
        vehicle = world.try_spawn_actor(bp, sp)
        if vehicle:
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
    return vehicles

def update_persistent_boxes(detections):
    global persistent_boxes
    new_boxes = {}
    id_counter = 0

    # Add new detections
    for det in detections:
        *xyxy, conf, cls_id = det
        cls_id = int(cls_id)
        box = list(map(int, xyxy))
        new_boxes[id_counter] = {'box': box, 'class': cls_id, 'frames_left': BOX_LIFETIME}
        id_counter += 1

    # Decrease old boxes' frames_left
    for old_id in list(persistent_boxes.keys()):
        persistent_boxes[old_id]['frames_left'] -= 1
        if persistent_boxes[old_id]['frames_left'] <= 0:
            del persistent_boxes[old_id]

    # Merge new detections into persistent_boxes
    persistent_boxes.update(new_boxes)

def draw_boxes(frame):
    for box_data in persistent_boxes.values():
        x1, y1, x2, y2 = box_data['box']
        cls_id = box_data['class']
        color = class_colors.get(cls_id, (255, 255, 255))
        label = class_names.get(cls_id, f'class {cls_id}')
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    global frame
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world('Town10HD')
    world = client.get_world()

    set_night_weather(world)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    # Spawn ego vehicle at 5th spawn point (index 4)
    ego_bp = random.choice(blueprint_library.filter('vehicle.tesla.model3'))
    start_index = 3
    ego_vehicle = world.try_spawn_actor(ego_bp, spawn_points[start_index] if len(spawn_points) > start_index else spawn_points[0])

    if not ego_vehicle:
        print(" Failed to spawn ego vehicle.")
        return
    print(" Ego vehicle spawned.")

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 20.0)
    ego_vehicle.set_autopilot(True, traffic_manager.get_port())

    # Attach RGB camera
    cam_bp = blueprint_library.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '640')
    cam_bp.set_attribute('image_size_y', '640')
    cam_bp.set_attribute('fov', '90')
    cam_bp.set_attribute('sensor_tick', '0.05')
    cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=ego_vehicle)
    cam.listen(camera_callback)

    # Load YOLOv5
    model_path = 'C:/Users/subhr/Documents/Carladeploy/yolov5/best.pt'
    if not os.path.exists(model_path):
        print(f" Model not found: {model_path}")
        return
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.conf = 0.4

    # Spawn actors
    extra_vehicles = spawn_vehicles(world, blueprint_library, count=15)
    walkers, controllers = spawn_pedestrians(world, blueprint_library, client, count=30)
    print(f" Spawned {len(walkers)} pedestrians and {len(extra_vehicles)} vehicles.")

    timeout = time.time() + 10
    while frame is None:
        world.tick()
        if time.time() > timeout:
            print(" Camera did not start.")
            return

    print(" Running detection (ESC to quit)...")
    try:
        while True:
            world.tick()
            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()
            update_persistent_boxes(detections)
            draw_boxes(frame)

            cv2.imshow("YOLOv5 Detection - Stable Night", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        print(" Cleaning up...")
        cam.stop()
        cam.destroy()
        ego_vehicle.destroy()
        for v in extra_vehicles:
            v.destroy()
        for c in controllers:
            c.stop()
            c.destroy()
        for w in walkers:
            w.destroy()
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))
        cv2.destroyAllWindows()
        print(" Done.")

if __name__ == '__main__':
    main()

 

