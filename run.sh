export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m paddle.distributed.launch predict.py \
    --config configs/${base_model}/${model}.yml \
    --model_path ${save_dir}/best_model/model.pdparams \
    --image_path data/cityscapes/leftImg8bit/test \
    --save_dir ${save_dir} \
    --aug_pred \
    --scale ${scale} \
&& \
python convert_cityscapes_trainid2labelid.py \
 --root_dir ${save_dir}/pseudo_color_prediction
&& \
cd ${save_dir}/pseudo_color_prediction && \
zip -r convert_to_labelid.zip  convert_to_labelid/ && \
cd -

