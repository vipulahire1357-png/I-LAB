import argparse
import logging
import sys
import time

try:
    from src.camera_manager import CameraManager
    from src.rtsp_client import RTSPClient
    from src.frame_extractor import FrameExtractor
except ModuleNotFoundError:  # pragma: no cover
    from camera_manager import CameraManager
    from rtsp_client import RTSPClient
    from frame_extractor import FrameExtractor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger("main")
    
    parser = argparse.ArgumentParser(description="RTSP Camera Frame Extraction Module")
    parser.add_argument("--camera", type=str, help="Camera ID to connect to (e.g., A14-419)")
    parser.add_argument("--list-cameras", action="store_true", help="List available cameras in the configuration")
    parser.add_argument("--fps", type=float, help="Override the extraction FPS configured in YAML")
    
    args = parser.parse_args()
    
    try:
        manager = CameraManager()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    if args.list_cameras:
        print("\nAvailable cameras:\n")
        for cam_id, cam_name in manager.list_cameras():
            print(f"{cam_id} - {cam_name}")
        print()
        sys.exit(0)
        
    if not args.camera:
        logger.error("Please specify a camera using --camera <ID> or use --list-cameras")
        sys.exit(1)
        
    try:
        cam_config = manager.get_camera_config(args.camera)
    except ValueError as e:
        logger.error(str(e))
        print("\nAvailable cameras:")
        for cam_id, _ in manager.list_cameras():
            print(cam_id)
        sys.exit(1)
        
    logger.info(f"Selected camera: {args.camera}")
    logger.info(f"Camera name: {cam_config['name']}")
    
    # Global configs
    extract_config = manager.get_global_config("frame_extraction", {})
    if args.fps is not None:
        extract_config["fps"] = args.fps
        
    conn_config = manager.get_global_config("connection", {})
    output_config = manager.get_global_config("frame_output", {})
    
    # Initialize pipeline
    client = RTSPClient(args.camera, cam_config, conn_config)
    extractor = FrameExtractor(args.camera, client, extract_config, output_config)
    
    logger.info("Starting extraction pipeline. Press CTRL+C to stop.")
    extractor.start()
    
    try:
        # Mock Computer Vision Pipeline Consumer Loop
        while True:
            # Block until a frame is ready
            frame_data = extractor.get_frame(timeout=1.0)
            if frame_data:
                # In a real scenario, we pass this frame to the CV pipeline
                logger.info(f"CV Pipeline received frame from {frame_data.camera_id} at {frame_data.timestamp} (shape: {frame_data.frame.shape})")
                
                # simulate processing time
                # time.sleep(0.1) 
    except KeyboardInterrupt:
        logger.info("CTRL+C detected. Shutting down gracefully...")
    finally:
        extractor.stop()
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main()
