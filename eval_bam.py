from bam_poses.eval.evaluation import Evaluation
import bam_poses.eval.utils as eval_utils
from bam_poses.transforms.transforms import normalize, undo_normalization_to_seq
from model import AttModel
from utils.opt import Options

import torch
import numpy as np

TMP_DIR = "/home/ssg1002/BAM_cache"  # this path needs ~125GB of free storage

EVAL_DATASET = "D"  # choose the evaluation dataset
ev = Evaluation(
    dataset=EVAL_DATASET, 
    tmp_dir=TMP_DIR,
    n_in=50, 
    n_out=250,
    data_location="/home/ssg1002/Datasets/BAM/0_1_0",
    default_to_box
    =True
)
# Important! Creating the evaluation for the first time will take a considerable amount
# of time (potentially ~2h) - make sure to provide a TMP_DIR where you can store the
# data for future use!


option = Options.init_from_json("./checkpoint/main_bam_3d_in50_out25_ks10_dctn35/option.json")

in_features = option.opt.in_features  # 51
d_model = option.opt.d_model
kernel_size = option.opt.kernel_size

model = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=option.opt.num_stage, dct_n=option.opt.dct_n)
model.cuda()
model.eval()

ckpt = torch.load(option.opt.ckpt + "/ckpt_best.pth.tar")
model.load_state_dict(ckpt["state_dict"])

def fn(persons_in, masks_in, scene, frame, n_in, n_out, pids):
    """
    Callback for generating the results. Your model predicts the
    data in here.
    :param persons_in: {n_persons x n_in x 17 x 3}
    :param masks_in: {n_persons x n_in}
    :param scene: {bam_poses.data.scene.Scene}
    :param frame: {int}
    :param n_in: {int}
    :param n_out: {int}
    :param pids: {List[int]}
    """
    # note that we don't batch the data. Before passing to the
    # model you will have to "batch" your data:
    
    persons_out = []
    for person in persons_in:
        normalized_seq, normalization_params = normalize(person, -1, return_transform=True)
        person_out = []
        n_iters = n_out // option.opt.output_n
        for i in range(n_iters):
            person_in = torch.from_numpy(normalized_seq.reshape(n_in, -1)).unsqueeze(0).cuda()
            person_out_hat = model(person_in, output_n=option.opt.output_n, input_n=option.opt.input_n, itera=1)  # predict using your model

            person_out_hat = undo_normalization_to_seq(
                person_out_hat[0, -option.opt.output_n:].cpu().detach().numpy().reshape(-1, 17, 3),
                normalization_params[0],
                normalization_params[1]
            )
            person_out.append(person_out_hat)
            ip = undo_normalization_to_seq(normalized_seq[option.opt.output_n - option.opt.input_n:], normalization_params[0], normalization_params[1])
            person_in = np.concatenate((ip, person_out_hat), 0)
            normalized_seq, normalization_params = normalize(person_in, -1, return_transform=True)
        person_out = np.concatenate(person_out, 0)
        persons_out.append(person_out)
    persons_out_hat = np.array(persons_out)
    return persons_out_hat.astype(np.float64)  # note that we have to "unbatch"


# run the evaluation
result = ev.ndms(fn)

# save results to file
eval_utils.save_results(TMP_DIR + "/results_hisrepitself.pkl", result)