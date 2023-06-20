# Training
## GEC:
```shell
fairseq-train ${conll14_data_path} \
  --save-dir ${save_path} \
  --dual-training-for-deletion \
  --dual-training-for-insertion \
  --y0 source \
  --no-share-discriminator \
  --ddp-backend=no_c10d --fp16 \
  --task translation_lev \
  --criterion nat_loss \
  --arch levenshtein_transformer \
  --label-smoothing 0.1 \
  --attention-dropout 0.2 \
  --activation-dropout 0.2 \
  --dropout 0.3 \
  --noise random_delete \
  --share-all-embeddings \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 5e-4 --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 --warmup-updates 20000 \
  --max-update 80000 \
  --weight-decay 0.0 --clip-norm 0.1 \
  --max-tokens 16384 --update-freq 1 \
  --apply-bert-init \
  --no-progress-bar --log-format 'simple' --log-interval 100 \
  --fixed-validation-seed 7 \
  --seed 1 \
  --save-interval-updates 4000 \
  --no-epoch-checkpoints \
  --fp16-scale-tolerance 0.1
```

## MT:
```shell
fairseq-train ${wmt_data_path} \
  --save-dir ${save_path} \
  --no-share-discriminator \
  --dual-training-for-deletion \
  --y0 self_gen \
  --ddp-backend=no_c10d --fp16 \
  --task translation_lev \
  --criterion nat_loss \
  --arch levenshtein_transformer \
  --label-smoothing 0.1 \
  --attention-dropout 0.1 \
  --activation-dropout 0 \
  --dropout 0.1 \
  --noise random_delete \
  --share-all-embeddings \
  --skip-invalid-size-inputs-valid-test \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 5e-4 --lr-scheduler inverse_sqrt \
  --warmup-init-lr 1e-07 --warmup-updates 10000 \
  --max-update 70000 \
  --weight-decay 0.0 --clip-norm 0.1 \
  --max-tokens 20000 --update-freq 3 \
  --apply-bert-init \
  --no-progress-bar --log-format 'simple' --log-interval 100 \
  --fixed-validation-seed 7 \
  --seed 1 \
  --save-interval-updates 3500 \
  --no-epoch-checkpoints \
  --fp16-scale-tolerance 0.1 
```
In Fairseq, the number of tokens in a batch = GPU number * max_tokens * update_freq. If you have 8 GPUs, the above scripts will have approximating 480k tokens in a batch.

# Citation

Please cite as:

``` bibtex
@inproceedings{zheng2023towards,
  title={Towards a Unified Training for Levenshtein Transformer},
  author={Zheng, Kangjie and Wang, Longyue and Wang, Zhihao and Chen, Binqi and Zhang, Ming and Tu, Zhaopeng},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
