########################################################################
# import default python-library
########################################################################
import os
import glob
import csv
import re
import itertools
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
from sklearn import metrics
# original lib
import common as com
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################


########################################################################
# def
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def test_id_list_generator(target_dir,
                           dir_name="test",
                           ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    test_dir_name : str (default="test")
        directory name the test data located in
    ext : str (default="wav)
        file name extension of audio files

    return :
        id_list : list [ str ]
            id list extracted from the test file name
    """
    # create test files
    chk_test_id_list = sorted(glob.glob("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext)))
    # Extract id
    id_list = sorted(list(set(itertools.chain.from_iterable([re.findall('id_[0-9][0-9]', ext_id) for ext_id in chk_test_id_list]))))
    return id_list


def test_files_list_generator(target_dir,
                              id_name,
                              dir_name="test",
                              prefix_nomal="normal",
                              prefix_anomaly="anomaly",
                              ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    test_dir_name : str (default="test")
        director name the test data located in
    normal_file_name : str (default="normal")
        normal data file name
    anomaly_file_name : str (default="anomaly")
        anomaly data file name
    ext : str (default="wav")
        file name extension of audio files

    return :
        if active type the development :
            test_files : list [ str ]
                file list for evaluation
            test_labels : list [ boolean ]
                label info. list for evaluation
                * normal/anomaly = 0/1
        if active type the evaluation :
            test_files : list [ str ]
                file list for evaluation
    """
    com.logger.info("target_dir : {}".format(target_dir+"/"+id_name))
    # Development
    if mode:
        normal_files = sorted(glob.glob("{dir}/{dir_name}/{prefix_nomal}_{id_name}*.{ext}".format(dir=target_dir, dir_name=dir_name, prefix_nomal=prefix_nomal, id_name=id_name, ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir, dir_name=dir_name, prefix_anomaly=prefix_anomaly, id_name=id_name, ext=ext)))
        anormaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anormaly_labels), axis=0)
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0: com.logger.exception(f'{"no_wav_data!!"}')
        print("\n========================================")

    # Evaluation
    else:
        files = sorted(glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir, dir_name=dir_name, id_name=id_name, ext=ext)))
        labels = None
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0: com.logger.exception(f'{"no_wav_data!!"}')
        print("\n=========================================")

    return files, labels
########################################################################


########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # mode check
    # mode:True == Development
    # mode:False == Evaluation
    mode = com.command_line_chk()
    if mode is None:
        exit(-1)
    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)
    # load base_directory
    dirs = com.select_dirs(param=param, mode=mode)
    # setup the result
    result_file = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
    result = []

    # loop of the base directory
    for num, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname=target_dir, num=num + 1, total=len(dirs)))

        machine_type = os.path.split(target_dir)[1]
        id_list = test_id_list_generator(target_dir)

        # setup model path
        model_file = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"], machine_type=machine_type)

        print("============== MODEL LOAD ==============")
        model = com.keras_model(param["feature"]["n_mels"] * param["feature"]["frames"])
        model.summary()

        # load model file
        if os.path.exists(model_file):
            model.load_weights(model_file)
        else:
            com.logger.error("{} model not found ".format(machine_type))
            exit(-1)

        if mode:
            # results by type
            result.append([machine_type])
            result.append(["id", "AUC", "pAUC"])
            result_machine_ave = []

        for id_str in id_list:
            # load test file
            test_files, y_true = test_files_list_generator(target_dir, id_str)

            # setup anomaly score file path
            result_anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_num}.csv"\
                                       .format(result=param["result_directory"],
                                               machine_type=machine_type,
                                               id_num=id_str)
            result_anomaly_score_list = []
            # test start
            print("\n============== TEST ==============")
            y_pred = [0. for k in test_files]
            for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
                try:
                    data = com.file_to_vector_array(file_path,
                                                      n_mels=param["feature"]["n_mels"],
                                                      frames=param["feature"]["frames"],
                                                      n_fft=param["feature"]["n_fft"],
                                                      hop_length=param["feature"]["hop_length"],
                                                      power=param["feature"]["power"])
                    error = numpy.mean(numpy.square(data - model.predict(data)), axis=1)
                    y_pred[file_idx] = numpy.mean(error)
                    result_anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                except:
                    com.logger.error("File broken!!: {}".format(file_path))

            # save anomaly_score
            save_csv(save_file_path=result_anomaly_score_csv, save_data=result_anomaly_score_list)
            com.logger.info("anomaly score result ->  {}".format(result_anomaly_score_csv))

            if mode:
                # create auc and p_auc data list
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                result.append([id_str.split("_", 1)[1], auc, p_auc])
                result_machine_ave.append([auc, p_auc])
                com.logger.info("AUC : {}".format(auc))
                com.logger.info("pAUC : {}".format(p_auc))
        if mode:
            # create average data
            result_machine_ave = numpy.mean(numpy.array(result_machine_ave, dtype=float), axis=0)
            average = ["Average"]
            average.extend(list(result_machine_ave))
            result.append(average)
            result.append([])
    if mode:
        # output results
        com.logger.info("AUC and pAUC results -> {}".format(result_file))
        save_csv(save_file_path=result_file, save_data=result)
    print("\n============ END TEST ============")