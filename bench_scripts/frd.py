import sys
import tensorflow as tf
import os
from tensorflow.python.platform import gfile

DATA_DIR = os.path.expanduser("/home/nahmad/t2t/data") # This folder contain the data
TMP_DIR = os.path.expanduser("/home/nahmad/t2t/tmp")
TRAIN_DIR = os.path.expanduser("/home/nahmad/t2t/train") # This folder contain the model
EXPORT_DIR = os.path.expanduser("/home/nahmad/t2t/export") # This folder contain the exported model for production
TRANSLATIONS_DIR = os.path.expanduser("/home/nahmad/t2t/translation") # This folder contain  all translated sequence
EVENT_DIR = os.path.expanduser("/home/nahmad/t2t/event") # Test the BLEU score
USR_DIR = os.path.expanduser("/home/nahmad/t2t/user") # This folder contains our data that we want to add
 
gfile.MakeDirs(DATA_DIR)
gfile.MakeDirs(TMP_DIR)
gfile.MakeDirs(TRAIN_DIR)
gfile.MakeDirs(EXPORT_DIR)
gfile.MakeDirs(TRANSLATIONS_DIR)
gfile.MakeDirs(EVENT_DIR)
gfile.MakeDirs(USR_DIR)

import sys
sys.path.insert(0,'./models') #replace this path by your own
from tensor2tensor.utils import registry
from tensor2tensor import problems

PROBLEM = "translate_ende_wmt32k"
MODEL = sys.argv[1] # mtf_transformer
HPARAMS = sys.argv[2] ##"mtf_transformer_52_2048_4096_model_4_batch_2"#"transformer_52_2048_4096" #

problems.available() #Show all problems
registry.list_models() #Show all registered models

#or
print("gen started")
t2t_problem = problems.problem(PROBLEM)
#t2t_problem.generate_data(DATA_DIR, TMP_DIR)
print("gen is done")

train_steps = 200 # Total number of train  steps for all Epochs
eval_steps = 110 # Number of steps to perform for each evaluation
batch_size = int(sys.argv[3])
save_checkpoints_steps = 1000000
ALPHA = 0.1
schedule = "train"

from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor.utils import registry
from tensor2tensor import models
from tensor2tensor import problems

# Init Hparams object from T2T Problem
hparams = create_hparams(HPARAMS)

# Make Changes to Hparams
hparams.batch_size = batch_size
hparams.learning_rate = ALPHA
#hparams.max_length = 256

# Can see all Hparams with code below
#print(json.loads(hparams.to_json())
gpus = 1
if len(sys.argv) >= 5:
  gpus = int(sys.argv[4])

RUN_CONFIG = create_run_config(
      model_dir=TRAIN_DIR,
      model_name=MODEL,
      save_checkpoints_steps= save_checkpoints_steps,
      num_gpus=gpus,
      gpu_mem_fraction=0.97
)

tensorflow_exp_fn = create_experiment(
        run_config=RUN_CONFIG,
        hparams=hparams,
        model_name=MODEL,
        problem_name=PROBLEM,
        data_dir=DATA_DIR, 
        train_steps=train_steps, 
        eval_steps=eval_steps,
        #use_xla=True # For acceleration
    ) 

tensorflow_exp_fn.train_and_evaluate()