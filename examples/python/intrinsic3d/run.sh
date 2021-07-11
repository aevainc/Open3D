DATAPATH=$1
python preprocess.py $DATAPATH
python colorize_coarse.py $DATAPATH
python colorize_fine.py $DATAPATH
python select_voxels.py
python associate_kfs.py $DATAPATH
python refine.py $DATAPATH
python sv_lighting.py $DATAPATH
python refine.py $DATAPATH
python colorize_coarse.py $DATAPATH --input colored_voxels_refined.npz --output colored_voxels_refined_coarse.npz
python colorize_fine.py $DATAPATH --input colored_voxels_refined_coarse.npz --output colored_voxels_refined_fine.npz
python postprocess.py $DATAPATH --input colored_voxels_refined_fine.npz
