# Before running
1. Use youtube_trainer to download and process data
2. Use depth_pro to get depth images
3. Use SAM2 to get mask images
4. Use GeoCalib to get CAM_K
5. Adjust shorter_side in the run_custom.py
6. Put all data in the demo_data folder

# Run with docker

1. `cd docker`
2. `bash run_container.sh`
3. Inside the docker, find the BundleSDF directory (~/Documents/BundleSDF)
4. `bash build.sh`
5. `cd mycuda`
6. `rm -rf build/ *.so`
7. `python3 setup.py build_ext --inplace`
8. `python3 run_batch.py --mode run_video --video_dir /home/BundleSDF/demo/ironing/estimation_results --out_folder /home/BundleSDF/demo/ironing/bundlesdf --timeout 900 --use_gui 0`