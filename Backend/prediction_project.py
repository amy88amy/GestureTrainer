import json
import pickle
from collections import defaultdict
import logging
from math import floor

import pandas as pd

def prediction(json_obj):

    json_list = json.loads(json_obj)

    df_1 = pd.DataFrame.from_records(json_list[0])
    df_2 = pd.DataFrame.from_records(json_list[1])

    rows = df_1.shape[0] if df_1.shape[0]<df_2.shape[0] else df_2.shape[0]

    df_1 = df_1.head(rows)
    df_2 = df_2.head(rows)

    curr_df = pd.concat([df_1, df_2], axis = 1)

    del_len=int(len(curr_df)*0.1)
    curr_df = curr_df.iloc[int(del_len/2):-int(del_len/2)]

    window_size = 30
    stride = 15

    final_column_names = ['max', 'may', 'maz', 'mgx', 'mgy', 'mgz', 'sax', 'say', 'saz', 'sgx', 'sgy', 'sgz']

    data = pd.DataFrame(columns=final_column_names)
    in_size = curr_df.shape[0]
    out_size = floor((in_size - window_size)/stride) + 1

    for i in range(int(out_size)):
        new_mean_row = curr_df.iloc[i*stride : (i*stride) + window_size].mean()
        new_sd_row = curr_df.iloc[i*stride : (i*stride) + window_size].std()
        final_row = new_mean_row.append(new_sd_row, ignore_index = True)
        final_row.index = final_column_names
        data = data.append(final_row, ignore_index=True)

    if data.shape[0] <= 7:
        logging.info("Not enough sensor data")
        return "Not enough sensor data"

    model = pickle.load(open('HARClassifier.sav','rb'))
    pred=model.predict(data)
    logging.info(pred)
    logging.info(len(pred))

    d = defaultdict(int)
    for i in pred:
        d[i] += 1
    label = max(d.items(), key=lambda x: x[1])

    logging.info(label[1])
    logging.info(label[0])
    logging.info((1.0*label[1])/len(pred))
    if ((1.0*label[1])/len(pred)) <= 0.5:
        return "No activity detected"

    logging.info(label)

    return label[0]

