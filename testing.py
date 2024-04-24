import pandas as pd
import mariadb
import sys
import numpy as np
from tqdm import tqdm


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches


def get_text_for_id(text_id, cur):
    textq = ["SELECT text FROM textreuse_sources ts",
             "INNER JOIN textreuse_ids ti ON ts.trs_id = ti.trs_id",
             'WHERE ti.manifestation_id = "' + text_id + '"']
    query_str = " ".join(textq)
    cur.execute(query_str)
    res = cur.fetchall()
    if len(res) > 1:
        print("More than one text for id: " + text_id)
        return None
    elif len(res) < 1:
        print("No text found for id: " + text_id)
        return None
    return res[0][0]


def get_tr_pairs_for_ecco_id_df(ecco_id, cur):
    query_str = " ".join([
        "SELECT",
        # "dst_trs_id as target_trs_id,",
        "dst_trs_start as target_start,",
        "dst_trs_end as target_end,",
        # "src_trs_id,",
        "src_trs_start,",
        "src_trs_end,",
        "src_ti.manifestation_id as src_manifestation_id,",
        "src_ei.edition_id as src_edition_id,",
        "src_mpd.publication_date as src_publication_date,",
        "src_mt.title as src_title",
        "FROM reception_edges_between_books_denorm",
        "INNER JOIN textreuse_ids target_ti ON target_ti.trs_id = dst_trs_id",
        "INNER JOIN textreuse_ids src_ti ON src_ti.trs_id = src_trs_id",
        "INNER JOIN manifestation_ids src_mi ON src_mi.manifestation_id = src_ti.manifestation_id",
        "INNER JOIN manifestation_publication_date src_mpd ON src_mi.manifestation_id_i = src_mpd.manifestation_id_i",
        "INNER JOIN manifestation_title src_mt ON src_mt.manifestation_id_i = src_mi.manifestation_id_i",
        "INNER JOIN edition_mapping src_em ON src_em.manifestation_id_i = src_mi.manifestation_id_i",
        "INNER JOIN edition_ids src_ei ON src_ei.edition_id_i = src_em.edition_id_i",
        "INNER JOIN edition_authors src_ea ON src_ea.edition_id_i = src_em.edition_id_i",
        "LEFT JOIN actor_ids src_ai ON src_ai.actor_id_i = src_ea.actor_id_i",
        'WHERE target_ti.manifestation_id = "' + ecco_id + '"'])
    cur.execute(query_str)
    res = pd.DataFrame(cur, columns=['dst_start', 'dst_end', 'src_start', 'src_end', 'src_ecco_id', 'src_estc_id',
                                     'src_pub_date', 'src_title'])
    return res


try:
    conn = mariadb.connect(
        user="sds-root",
        password="ryfjok-qivfub-quhXe4",
        host="vm4319.kaj.pouta.csc.fi",
        port=3306,
        database="hpc-hd-newspapers"
    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

# Get Cursor
cur = conn.cursor()

# Hume publications
query_str_hume = " ".join([
    "SELECT ec.estc_id, ec.publication_year, ea.actor_id, ea.name_unified, ec2.ecco_full_title, ec2.ecco_id, ec2.ecco_date_start FROM estc_core ec",
    "INNER JOIN estc_actor_links eal ON ec.estc_id = eal.estc_id",
    "INNER JOIN estc_actors ea ON eal.actor_id = ea.actor_id",
    "INNER JOIN ecco_core ec2 ON ec2.estc_id = ec.estc_id",
    "WHERE ea.actor_id = '49226972'"])
cur.execute(query_str_hume)
hume_titles = pd.DataFrame(cur, columns=['estc_id', 'estc_publication_year', 'actor_id', 'actor_name_unified',
                                         'ecco_full_title', 'ecco_id', 'ecco_date_start'])

hume_hoe_titles = hume_titles[hume_titles.ecco_full_title.str.lower().str.contains("history") &
                              (hume_titles.ecco_full_title.str.lower().str.contains("england") |
                              hume_titles.ecco_full_title.str.lower().str.contains("great britain"))]


# testres = get_tr_pairs_for_ecco_id_df("0145000201", cur)


ecco_id = "0145000201"
testres2 = get_sources_and_indices_for_eccoid(ecco_id=ecco_id, cur=cur)



with open('testout.txt', 'w') as outf:
    outf.writelines(list(set([row['src_id'] + '\n' for row in testres2])))


# Hume 1778 History edition in 8 volumes: estc_id T82472
# Rapin, T144219;
# Carte, T101742;
# Guthrie, T138171.

# 1. get all text reuse pieces for ecco_id
# Hume vol 1: 0145000201



def get_text_for_ecco_id(ecco_id, cur):
    query_str = " ".join([
        'SELECT manifestation_id, text FROM textreuse_sources ts',
        'INNER JOIN textreuse_manifestation_mapping tmm ON ts.trs_id = tmm.trs_id',
        'INNER JOIN manifestation_ids mi',
        'ON tmm.manifestation_id_i = mi.manifestation_id_i',
        'WHERE manifestation_id = "' + ecco_id + '"'])
    cur.execute(query_str)
    res = list()
    for item in cur:
        res.append({
            'ecco_id': item[0],
            'text': item[1],
        })
    return res


thistext = get_text_for_ecco_id("0000100100", cur)


import json
from glob import glob


def get_offset_data(offset_data_path):
    offset_data = dict()
    offset_files = glob(offset_data_path + "/tr_offset*.json")
    for file_path in offset_files:
        with open (file_path, 'r') as jsonf:
            offsets = json.load(jsonf)
            for doc_id, off_con in offsets.items():
                off_l = list()
                for k, v in off_con.items():
                    thisline = dict()
                    thisline['doc_index'] = int(k)
                    thisline['header'] = v['header']
                    off_l.append(thisline)
                off_l = sorted(off_l, key=lambda d: d['doc_index'])
                offset_data[doc_id] = off_l
    return offset_data


offset_data_path = "../data_input_text_reuse/data/work/tr_off/"
offset_data = get_offset_data(offset_data_path)


with open ("../data_input_text_reuse/data/work/tr_off/tr_offset_10000.json", 'r') as jsonf:
    offsets = json.load(jsonf)
    off_d = dict()
    for doc_id, off_con in offsets.items():
        off_l = list()
        for k, v in off_con.items():
            thisline = dict()
            thisline['doc_index'] = int(k)
            thisline['header'] = v['header']
            off_l.append(thisline)
        off_l = sorted(off_l, key=lambda d: d['doc_index'])
        off_d[doc_id] = off_l

off_df = pd.DataFrame.from_dict(off_l)


def print_clips_and_headers(ecco_id, offsets_dict, cur):
    this_text = get_text_for_ecco_id(ecco_id, cur)[0]['text']
    this_offsets = offsets_dict[ecco_id]
    print(ecco_id)
    print()
    for item in this_offsets:
        print('------')
        print(item['header'].strip())
        print('------')
        print(this_text[item['doc_index']:item['doc_index']+200])
        print()


print_clips_and_headers('0000100200', off_d, cur)

'0063601100'

this_text = get_text_for_ecco_id('0063601100', cur)

# get section titles and their position in text
#
#
#
# this_id = "0126700101"
# this_id = "1446300105"
# this_id = "0971100108"
# this_id = "0582400106"
# this_id = "0162900301"
# test_text = get_text_for_id(this_id, cur)
#
#
# potential_ids = list()
# for text_id in filtered.ecco_id.to_list():
#     test_text = get_text_for_id(text_id, cur)
#     if len(list(find_all(test_text, '# '))) > 1:
#         print("Found: " + text_id)
#         potential_ids.append(text_id)
#
#
# this_id = "1131700108"
#
#
# def print_clips_from_text(text_id, cur):
#     test_text = get_text_for_id(text_id, cur)
#     offsets = list(find_all(test_text, '# '))
#     for offset in offsets:
#         print(test_text[offset-10:offset+100])
#         print()
#
#
# print_clips_from_text(potential_ids[26], cur)
