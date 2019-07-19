## Pretraining with XLNet
`cd xlnet/`
### 1. 先训练spm模型
```bash
spm_train \
	--input=$INPUT \
	--model_prefix=sp10m.cased.v3 \
	--vocab_size=32000 \
	--character_coverage=0.99995 \
	--model_type=unigram \
	--control_symbols=<cls>,<sep>,<pad>,<mask>,<eod> \
	--user_defined_symbols=<eop>,.,(,),",-,–,£,€ \
	--shuffle_input_sentence \
	--input_sentence_size=10000000
```

Special symbols are used, including `control_symbols` and `user_defined_symbols`. We use `<eop>` and `<eod>` to denote End of Paragraph and End of Document respectively.


### 2. 再转成tfrecord
Refer to `train.py` for pretraining on TPUs and `train_gpu.py` for pretraining on GPUs. First we need to preprocess the text data into tfrecords.
```shell
python data_utils.py \
	--bsz_per_host=32 \
	--num_core_per_host=16 \
	--seq_len=512 \
	--reuse_len=256 \
	--input_glob=*.txt \
	--save_dir=${SAVE_DIR} \
	--num_passes=20 \
	--bi_data=True \
	--sp_path=spiece.model \
	--mask_alpha=6 \
	--mask_beta=1 \
	--num_predict=85
```

where `input_glob` defines all input text files, `save_dir` is the output directory for tfrecords, and `sp_path` is a [Sentence Piece](https://github.com/google/sentencepiece) model. Here is our script to train the Sentence Piece model

The input text files to `data_utils.py` must use the following format:
* Each line is a sentence.
* An empty line means End of Document.
* (Optional) If one also wants to model paragraph structures, `<eop>` can be inserted at the end of certain lines (without any space) to indicate that the corresponding sentence ends a paragraph.

For example, the text input file could be:
```
This is the first sentence.
This is the second sentence and also the end of the paragraph.<eop>
Another paragraph.

Another document starts here.
```

### 3. 最后开始训练
After preprocessing, we are ready to pretrain an XLNet. Below are the hyperparameters used for pretraining XLNet-Large:

```shell
python train.py
  --corpus_info_path=$DATA/corpus_info.json \
  --record_info_dir=$DATA/tfrecords \
  --train_batch_size=2048 \
  --seq_len=512 \
  --reuse_len=256 \
  --perm_size=256 \
  --n_layer=24 \
  --d_model=1024 \
  --d_embed=1024 \
  --n_head=16 \
  --d_head=64 \
  --d_inner=4096 \
  --untie_r=True \
  --mask_alpha=6 \
  --mask_beta=1 \
  --num_predict=85
```

where we only list the most important flags and the other flags could be adjusted based on specific use cases.
