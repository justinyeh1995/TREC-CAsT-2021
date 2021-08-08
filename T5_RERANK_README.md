### Start new VM in Google Cloud
```sh
gcloud compute --project=doctttttquery instances create my-vm --zone=us-central1-b --machine-type=n1-standard-4 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=230744092782-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=debian-10-buster-v20210721 --image-project=debian-cloud --boot-disk-size=200GB --boot-disk-type=pd-standard --boot-disk-device-name=my-vm --reservation-affinity=any
```
### T5 Installation

To install the T5 package, simply run:

```sh
pip install t5[gcp]
```

### Setting up TPUs on GCP

You will first need to launch a Virtual Machine (VM) on Google Cloud. Details about launching the VM can be found at the [Google Cloud Documentation](https://cloud.google.com/compute/docs/instances/create-start-instance).

In order to run training or eval on Cloud TPUs, you must set up the following variables based on your project, zone and GCS bucket appropriately. Please refer to the [Cloud TPU Quickstart](https://cloud.google.com/tpu/docs/quickstart) guide for more details.

```sh
export PROJECT=your_project_name
export ZONE=your_project_zone
export BUCKET=gs://yourbucket/
export TPU_NAME=t5-tpu
export TPU_SIZE=v3-8
export DATA_DIR="${BUCKET}/your_data_dir"
export MODEL_DIR="${BUCKET}/your_model_dir"
```

Please use the following command to create a TPU device in the Cloud VM.

```sh
ctpu up --name=$TPU_NAME --project=$PROJECT --zone=$ZONE --tpu-size=$TPU_SIZE \
        --tpu-only --noconf
```

### Start a TPU.

```sh
ctpu up --name=my-tpu --project=doctttttquery --zone=us-central1-b \
    --tpu-size=v3-8  --tpu-only  --tf-version=2.5.0  --noconf  \
    --preemptible

```

### Install T5

```
sudo apt-get install gcc --yes

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh

exec bash

conda create --name py36 python=3.6
conda activate py36
conda install -c conda-forge httptools
conda install -c conda-forge jsonnet

pip install t5[gcp]
```

### IMPORTANT HACK:
We hack the t5 model to output probabilities instead of tokens.

Once installation is complete, the files 'transformer.py' and 'utils.py' should be put in:
/your_python_path/lib/python3.6/site-packages/mesh_tensorflow/transformer/

Example path:
/home/rodrigonogueira4/miniconda3/envs/py36/lib/python3.6/site-packages/mesh_tensorflow/transformer/

## Preparing data.
To train a T5 reranker, we need a TSV file that has the following format:
```
f'Query: {query} Document: {document} Relevant:\t{label}\n'
```
Where `query` is the query text, `document` is the document text, and `label` is the string "true" or "false" depending if the document is relevant or not to the query.

The DEV file is a txt file with one query + document text per line. It should have the following format:
```
f'Query: {query} Document: {document} Relevant:\n'
```
Note that it is not a TSV file nor it contains the label.

To create training and dev data for Robust04, you can modify the files that create data from MS MARCO:
`create_msmarco_tsv_training_pairs_t5_reranker.py` and `create_msmarco_tsv_dev_pairs_t5_reranker.py`.


## Start training
We can now start training. 

Note 1: Remember to change `your_bucket` accordingly.
Note 2: Put training tsv file in `gs://your_bucket/data/query_doc_pairs.train.tsv`.
Note 3: In the example below, the model will be trained by 1,100 iterations (1,001,000 - 999,900). Change this accordingly.
Note 4: Sometimes `t5_mesh_transformer` does not recognize the model checkpoint and it starts training from step #0. If you see that, stop training as your model is being trained from scratch!

conda activate py36
nohup t5_mesh_transformer  \
  --tpu="my_tpu" \
  --gcp_project="gin-project-261821" \
  --tpu_zone="us-central1-b" \
  --model_dir="gs://your_bucket/model" \
  --gin_param="init_checkpoint = 'gs://t5-data/pretrained_models/base/model.ckpt-999900'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://your_bucket/data/query_doc_pairs.train.tsv'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1001000" \
  --gin_param="tokens_per_batch = 131072" \
  >> out.log 2>&1 &


## Inference
We will now predict a score for each query-doc pair.
Note 1: The input file here is a txt with one query + document per line. The output will be the predicted label ("true" or "false") and the probability of being true (i.e., the document being relevant to the query).
Note 2: Remember to split the input file into files of 1M lines. Otherwise, TensorFlow will not be able to load it entirely into memory.
Note 3: 

```
conda activate py36

for ITER in {000..006}; do
  echo "Running iter: $ITER" >> out.log_eval.log
  nohup t5_mesh_transformer \
    --tpu="my_tpu" \
    --gcp_project="gin-project-261821" \
    --tpu_zone="us-central1-b" \
    --model_dir="gs://your_bucket/model" \
    --gin_file="gs://t5-data_v2/pretrained_models/base/operative_config.gin" \
    --gin_file="infer.gin" \
    --gin_file="beam_search.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
    --gin_param="infer_checkpoint_step = 1001000" \
    --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 4}" \
    --gin_param="Bitransformer.decode.max_decode_length = 4" \
    --gin_param="input_filename = 'gs://your_bucket/doc2query/data/query_doc_pairs.dev.small.txt${ITER}'" \
    --gin_param="output_filename = 'gs://your_bucket/doc2query/data/predictions_dev.txt${ITER}'" \
    --gin_param="tokens_per_batch = 131072" \
    --gin_param="Bitransformer.decode.beam_size = 1" \
    --gin_param="Bitransformer.decode.temperature = 0.0" \
    --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" \
    >> out.log 2>&1
done &

# Converting run to the TREC format.
We now need to create a TREC-formatted run file, which will contain the predicted probabilities, query ids, and doc ids. The script below does that but it outputs MS MARCO-formatted runs. It should be easy to modify it to output TREC-formatted runs. 
`convert_run_from_t5_to_msmarco_format.py`


