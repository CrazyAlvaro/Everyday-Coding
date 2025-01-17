from tqdm import tqdm


############################################################
## MACRO DEFINITION
############################################################

_LABEL_RU = ["Car", "SUV", "Truck", "Special_truck"]
_LABEL_VRU = ["tricycle", "bicycle", "motorbike"]


############################################################
## module methods
############################################################
def _is_RU(ru_label):
    return ru_label in set(_LABEL_RU)

def _is_VRU(ru_label):
    return ru_label in set(_LABEL_VRU)

############################################################
## NIO atomic labeling system checking [1-35]
############################################################
def _check_RU_horizontal(df_timeframe, ru_id):
    """
    return dict{} represents RU_hor mapping
    """
    _RU_hor = {
        1:10
    }

    return _RU_hor

def _check_RU_vertical(df_timeframe, ru_id):
    """
    return dict{} represents RU_ver mapping
    """
    _RU_ver = {
        3:15
    }

    return _RU_ver

def _check_VRU(df_timeframe, ru_id):
    """
    return dict{} represents VRU mapping
    """
    _VRU = {
        6:20
    }

    return _VRU

def _construct_label(df_frame, _index, _RU_label):
    for _key, _val in _RU_label.items():
        assert int(_key) >= 1 and int(_key) <= 35, "[check_label-_construct_label] Atomic labeling system index range [1-35]"
        df_frame.loc[_index, 'ru'+str(_key)] = _val

def _timeframe_processing(df_ru, _index):
    _curr_row = df_ru.loc[_index]
    _curr_obj_id = _curr_row['obj_id']
    _ts_group = df_ru[df_ru['ts'] == _curr_row['ts']]

    _RU_hor = _check_RU_horizontal(_ts_group, _curr_obj_id)
    _RU_ver = _check_RU_vertical(_ts_group, _curr_obj_id)
    _VRU = _check_VRU(_ts_group, _curr_obj_id)

    return {**_RU_hor, **_RU_ver, **_VRU}

############################################################
## module entry function:
##      processing tracks dataframe to have atomic label system
############################################################
def check_surrounding_label(df_ru):
    """
    return dict{} represents VRU mapping
    """
    # Create new `pandas` methods which use `tqdm` progress
    # tqdm.pandas()
    # df_checked = df_ru.groupby('ts').progress_apply()

    # print('before out check {}'.format(df_ru.loc[0, 'ru1']))

    for _index in tqdm(df_ru.index, desc="Checking Labels"):
        _RU_labels = _timeframe_processing(df_ru, _index)
        _construct_label(df_ru, _index, _RU_labels)

    # print('after out check {}'.format(df_ru.loc[0, 'ru1']))

    return df_ru

############################################################
##
############################################################






