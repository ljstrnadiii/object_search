bash bin/download_mscoco.sh

bash bin/download_models.sh

python search.py \
    --dataset "data/datasets/mscoco/*/*.jpg" \
    --batch_size 32 \
    --n_parallel_pipeline 3 \
    --n_chunks 4
