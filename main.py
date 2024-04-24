import json
from glob import glob
import pandas as pd
import mariadb
import sys
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import groupby


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


def get_tr_pairs_for_ecco_id_simple(ecco_id, cur):
    query_str = " ".join([
        "SELECT",
        # "dst_trs_id as target_trs_id,",
        "dst_trs_start as target_start,",
        "dst_trs_end as target_end,",
        "src_ti.manifestation_id as src_manifestation_id",
        "FROM reception_edges_between_books_denorm",
        "INNER JOIN textreuse_ids target_ti ON target_ti.trs_id = dst_trs_id",  # !
        "INNER JOIN textreuse_ids src_ti ON src_ti.trs_id = src_trs_id",  # !
        "INNER JOIN manifestation_ids src_mi ON src_mi.manifestation_id = src_ti.manifestation_id",  # !
        'WHERE target_ti.manifestation_id = "' + ecco_id + '"'])
    cur.execute(query_str)
    res = list()
    for item in cur:
        res.append({
            'dst_start': item[0],
            'dst_end': item[1],
            'src_id': item[2]
        })
    return res


def get_db_cursor():
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
    return cur


# Get Cursor
cur = get_db_cursor()

# Hume publications
hume_actor_id = '49226972'
query_str_hume = " ".join([
    "SELECT ec.estc_id, ec.publication_year, ea.actor_id, ea.name_unified, ec2.ecco_full_title, ec2.ecco_id, ec2.ecco_date_start FROM estc_core ec",
    "INNER JOIN estc_actor_links eal ON ec.estc_id = eal.estc_id",
    "INNER JOIN estc_actors ea ON eal.actor_id = ea.actor_id",
    "INNER JOIN ecco_core ec2 ON ec2.estc_id = ec.estc_id",
    "WHERE ea.actor_id = " + hume_actor_id])
cur.execute(query_str_hume)
hume_titles = pd.DataFrame(cur, columns=['estc_id', 'estc_publication_year', 'actor_id', 'actor_name_unified',
                                         'ecco_full_title', 'ecco_id', 'ecco_date_start'])

hume_hoe_titles = hume_titles[hume_titles.ecco_full_title.str.lower().str.contains("history") &
                              (hume_titles.ecco_full_title.str.lower().str.contains("england") |
                              hume_titles.ecco_full_title.str.lower().str.contains("great britain"))]


def get_sources_and_indices_for_eccoid(ecco_id, cur):
    query_str = " ".join([
        "SELECT",
        "dst_trs_start AS target_start,",
        "dst_trs_end AS target_end,",
        "src_ti.manifestation_id AS src_manifestation_id,",
        "src_ai.actor_id AS src_actor_id,",
        "src_mpd.publication_date as src_pub_date",
        "FROM reception_edges_between_books_denorm",
        "INNER JOIN textreuse_ids target_ti ON target_ti.trs_id = dst_trs_id",  # !
        "INNER JOIN textreuse_ids src_ti ON src_ti.trs_id = src_trs_id",  # !
        "INNER JOIN manifestation_ids src_mi ON src_mi.manifestation_id = src_ti.manifestation_id",  # !
        "INNER JOIN edition_mapping src_em ON src_em.manifestation_id_i = src_mi.manifestation_id_i",
        "LEFT JOIN edition_authors src_ea ON src_ea.edition_id_i = src_em.edition_id_i",
        "LEFT JOIN actor_ids src_ai ON src_ai.actor_id_i = src_ea.actor_id_i",
        "LEFT JOIN manifestation_publication_date src_mpd ON src_mpd.manifestation_id_i = src_mi.manifestation_id_i",
        'WHERE target_ti.manifestation_id = "' + ecco_id + '"'])
    cur.execute(query_str)
    res = list()
    for item in cur:
        res.append({
            'dst_start': item[0],
            'dst_end': item[1],
            'src_item_id': item[2],
            'src_author_id': item[3],
            'src_publication_date': item[4]
        })
    return res


def merge_intervals(intervals):
    # Sort the array on the basis of start values of intervals.
    intervals.sort()
    stack = list()
    # insert first interval into stack
    stack.append(intervals[0])
    for i in intervals[1:]:
        # Check for overlapping interval,
        # if interval overlap
        if stack[-1][0] <= i[0] <= stack[-1][-1]:
            stack[-1][-1] = max(stack[-1][-1], i[-1])
        else:
            stack.append(i)
    return stack


def filter_author_sources_and_indices(source_data, filter_author):
    filtered = [item for item in source_data if str(item['src_author_id']) != filter_author]
    return filtered


def get_source_coverage(sourcedata, target_text):
    intervals = list()
    for item in sourcedata:
        intervals.append([item['dst_start'], item['dst_end']])
    merged_intervals = merge_intervals(intervals)
    total_merged_length = sum([item[1]-item[0] for item in merged_intervals])
    coverage_ratio = total_merged_length/len(target_text)
    return {'ratio': coverage_ratio, 'sources_len': total_merged_length, 'target_len': len(target_text)}


actor_id = '49226972'  # David Hume
ecco_id = "0145000201"
testres2 = get_sources_and_indices_for_eccoid(ecco_id=ecco_id, cur=cur)
testdf = pd.DataFrame(testres2)
testtext = get_text_for_ecco_id(ecco_id, cur)[0]['text']
filtered_testdata = filter_author_sources_and_indices(testres2, hume_actor_id)
test_coverage = get_source_coverage(filtered_testdata, testtext)


def get_filtered_source_coverage_for_id(ecco_id, actor_id, cur):
    source_fragments = get_sources_and_indices_for_eccoid(ecco_id=ecco_id, cur=cur)
    target_text = get_text_for_ecco_id(ecco_id, cur)[0]['text']
    filtered_fragments = filter_author_sources_and_indices(source_fragments, actor_id)
    source_coverage = get_source_coverage(filtered_fragments, target_text)
    return source_coverage


hume_coverage = get_filtered_source_coverage_for_id(ecco_id="0145000201", actor_id='49226972', cur=cur)


def get_source_coverage_for_multivol(ecco_ids, actor_id, cur):
    results = dict()
    total_target_len = 0
    total_source_len = 0
    for ecco_id in ecco_ids:
        this_coverage = get_filtered_source_coverage_for_id(ecco_id=ecco_id, actor_id=actor_id, cur=cur)
        total_target_len += this_coverage['target_len']
        total_source_len += this_coverage['sources_len']
        results[ecco_id] = this_coverage
    total_ratio = total_source_len / total_target_len
    results['total_ratio'] = total_ratio
    results['total_source_len'] = total_source_len
    results['total_target_len'] = total_target_len
    return results


def write_json_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


hume_actor_id = '49226972'  # David Hume
# hume_vols = ["0145000202", "0145100105", "0145200108", "0145100106", "0145100103", "0145100104",
#              "0145000201", "0145100107"]
hume_vols = ['0162900301', '0162900302', '0429000101', '0429000102', '0156400400', '0162200200']
hume_coverage = get_source_coverage_for_multivol(hume_vols, hume_actor_id, cur)
write_json_results(hume_coverage, 'data/results/hume_coverage.json')

rapin_actor_id = "59111095"
rapin_vols = ["1393700102", "1394100115", "1394000111", "1393800105", "1394000110", "1394100113", "1394100114",
              "1393700103", "1394000112", "1393700101", "1393800106", "1393900109", "1393800104", "1393900108",
              "1393900107"]
rapin_coverage = get_source_coverage_for_multivol(rapin_vols, rapin_actor_id, cur)
write_json_results(rapin_coverage, 'data/results/rapin_coverage.json')

carte_actor_id = "34794161"
carte_vols = ["0018500104", "0018400102", "0018400101", "0018500103"]
carte_coverage = get_source_coverage_for_multivol(carte_vols, carte_actor_id, cur)
write_json_results(carte_coverage, 'data/results/carte_coverage.json')

guthrie_actor_id = "100194070"
guthrie_vols = ["1679100104", "1679000101", "1679100103"]
guthrie_coverage = get_source_coverage_for_multivol(guthrie_vols, guthrie_actor_id, cur)
write_json_results(guthrie_coverage, 'data/results/guthrie_coverage.json')

plot1_data = pd.DataFrame([{'author': 'Hume', 'reuse ratio': hume_coverage['total_ratio']},
              {'author': 'Rapin', 'reuse ratio': rapin_coverage['total_ratio']},
              {'author': 'Carte', 'reuse ratio': carte_coverage['total_ratio']},
              {'author': 'Guthrie', 'reuse ratio': guthrie_coverage['total_ratio']}])

sns.barplot(plot1_data, x='author', y='reuse ratio')
plt.savefig('plots/plot1.png')
plt.close('all')

plots2_to_5_data = pd.DataFrame([
    {'author': 'Hume', 'volume': '1', 'reuse ratio': hume_coverage['0145000201']['ratio']},
    {'author': 'Hume', 'volume': '2', 'reuse ratio': hume_coverage['0145000202']['ratio']},
    {'author': 'Hume', 'volume': '3', 'reuse ratio': hume_coverage['0145100103']['ratio']},
    {'author': 'Hume', 'volume': '4', 'reuse ratio': hume_coverage['0145100104']['ratio']},
    {'author': 'Hume', 'volume': '5', 'reuse ratio': hume_coverage['0145100105']['ratio']},
    {'author': 'Hume', 'volume': '6', 'reuse ratio': hume_coverage['0145100106']['ratio']},
    {'author': 'Hume', 'volume': '7', 'reuse ratio': hume_coverage['0145100107']['ratio']},
    {'author': 'Hume', 'volume': '8', 'reuse ratio': hume_coverage['0145200108']['ratio']},
    {'author': 'Rapin', 'volume': '1', 'reuse ratio': rapin_coverage['1393700101']['ratio']},
    {'author': 'Rapin', 'volume': '2', 'reuse ratio': rapin_coverage['1393700102']['ratio']},
    {'author': 'Rapin', 'volume': '3', 'reuse ratio': rapin_coverage['1393700103']['ratio']},
    {'author': 'Rapin', 'volume': '4', 'reuse ratio': rapin_coverage['1393800104']['ratio']},
    {'author': 'Rapin', 'volume': '5', 'reuse ratio': rapin_coverage['1393800105']['ratio']},
    {'author': 'Rapin', 'volume': '6', 'reuse ratio': rapin_coverage['1393800106']['ratio']},
    {'author': 'Rapin', 'volume': '7', 'reuse ratio': rapin_coverage['1393900107']['ratio']},
    {'author': 'Rapin', 'volume': '8', 'reuse ratio': rapin_coverage['1393900108']['ratio']},
    {'author': 'Rapin', 'volume': '9', 'reuse ratio': rapin_coverage['1393900109']['ratio']},
    {'author': 'Rapin', 'volume': '10', 'reuse ratio': rapin_coverage['1394000110']['ratio']},
    {'author': 'Rapin', 'volume': '11', 'reuse ratio': rapin_coverage['1394000111']['ratio']},
    {'author': 'Rapin', 'volume': '12', 'reuse ratio': rapin_coverage['1394000112']['ratio']},
    {'author': 'Rapin', 'volume': '13', 'reuse ratio': rapin_coverage['1394100113']['ratio']},
    {'author': 'Rapin', 'volume': '14', 'reuse ratio': rapin_coverage['1394100114']['ratio']},
    {'author': 'Rapin', 'volume': '15', 'reuse ratio': rapin_coverage['1394100115']['ratio']},
    {'author': 'Carte', 'volume': '1', 'reuse ratio': carte_coverage['0018400101']['ratio']},
    {'author': 'Carte', 'volume': '2', 'reuse ratio': carte_coverage['0018400102']['ratio']},
    {'author': 'Carte', 'volume': '3', 'reuse ratio': carte_coverage['0018500103']['ratio']},
    {'author': 'Carte', 'volume': '4', 'reuse ratio': carte_coverage['0018500104']['ratio']},
    {'author': 'Guthrie', 'volume': '1', 'reuse ratio': guthrie_coverage['1679000101']['ratio']},
    {'author': 'Guthrie', 'volume': '3', 'reuse ratio': guthrie_coverage['1679100103']['ratio']},
    {'author': 'Guthrie', 'volume': '4', 'reuse ratio': guthrie_coverage['1679100104']['ratio']},
])

for author in ['Hume', 'Rapin', 'Guthrie', 'Carte']:
    sns.barplot(plots2_to_5_data[plots2_to_5_data['author'] == author], x='volume', y='reuse ratio').set(
        title=author)
    plt.savefig('plots/plot_reuse_ratio_' + author + '.png')
    plt.close('all')




def get_combined_source_fragments_by_source_id(sourcedata):
    sourcedata.sort(key=lambda x: x['src_item_id'])
    grouped_results = dict()
    for k, v in groupby(sourcedata, key=lambda x: x['src_item_id']):
        grouped_results[k] = list(v)
    #
    ret_results = list()
    for k, v in grouped_results.items():
        intervals = list()
        for item in v:
            intervals.append([item['dst_start'], item['dst_end']])
            this_date = item['src_publication_date']
            this_author = item['src_author_id']
        merged_intervals = merge_intervals(intervals)
        for interval in merged_intervals:
            ret_results.append({'dst_start': interval[0], 'dst_end': interval[1], 'src_item_id': k,
                                'src_publication_date': this_date, 'src_author_id': this_author})
    return ret_results


def get_indices_overlap(early, late):
    indices_overlap_abs = len(range(max(early[0], late[0]), min(early[-1], late[-1]) + 1))
    indices_overlap = indices_overlap_abs / len(range(late[0], late[-1] + 1))
    return indices_overlap


# Test each item against all items already in the dataset. If they overlap at the threshold or greater,
# And they are later, do not add them to the new dataset. Otherwise, add them.
def get_source_fragments_merged_by_date(source_fragments, overlap_threshold):
    sorted_results = sorted(source_fragments, key=lambda x: x['src_publication_date'])
    date_merged = list()
    date_merged.append(sorted_results[0])
    for i in sorted_results[1:]:
        add_item = True
        for dm in date_merged:
            indices_overlap = get_indices_overlap([dm['dst_start'], dm['dst_end']], [i['dst_start'], i['dst_end']])
            if i['src_publication_date'] > dm['src_publication_date'] and indices_overlap >= overlap_threshold:
                add_item = False
        if add_item:
            date_merged.append(i)
    return date_merged

# testing the above
# for threshold_value in [item/100 for item in list(range(0,110, 10))]:
#     date_merged = get_source_fragments_merged_by_date(merged_source_fragments_per_ecco_id, threshold_value)
#     print(str(threshold_value) + ": " + str(len(date_merged)) + "/" + str(len(merged_source_fragments_per_ecco_id)))


def get_combined_source_fragments_by_author_id(sourcedata, target_text):
    for item in sourcedata:
        if item['src_author_id'] is None:
            item['src_author_id'] = 'NA'
    sourcedata.sort(key=lambda x: x['src_author_id'])
    grouped_results = dict()
    for k, v in groupby(sourcedata, key=lambda x: x['src_author_id']):
        grouped_results[k] = list(v)
    #
    ret_results = list()
    for k, v in grouped_results.items():
        intervals = list()
        for item in v:
            intervals.append([item['dst_start'], item['dst_end']])
        merged_intervals = merge_intervals(intervals)
        total_length = 0
        for interval in merged_intervals:
            total_length += interval[1] - interval[0]
        ret_results.append({'src_author_id': k, 'intervals': merged_intervals, 'combined_length': total_length,
                            'fragments': len(merged_intervals), 'portion_covered': total_length / len(target_text)})
        res_sorted = sorted(ret_results, key=lambda x: x['portion_covered'], reverse=True)
    return res_sorted


# def get_filtered_source_coverage_for_id_per_actor(ecco_id, actor_id, cur):
#     source_fragments = get_sources_and_indices_for_eccoid(ecco_id=ecco_id, cur=cur)
#     target_text = get_text_for_ecco_id(ecco_id, cur)[0]['text']
#     filtered_fragments = filter_author_sources_and_indices(source_fragments, actor_id)
#     actor_coverage = get_combined_source_fragments_by_author_id(filtered_fragments, target_text)
#     return actor_coverage


def get_filtered_source_coverage_for_id_per_actor(source_fragments, ecco_id, actor_id, cur):
    # source_fragments = get_sources_and_indices_for_eccoid(ecco_id=ecco_id, cur=cur)
    target_text = get_text_for_ecco_id(ecco_id, cur)[0]['text']
    filtered_fragments = filter_author_sources_and_indices(source_fragments, actor_id)
    actor_coverage = get_combined_source_fragments_by_author_id(filtered_fragments, target_text)
    return actor_coverage


def get_data_for_actor_id(actor_id, cur):
    query_str = " ".join([
        "SELECT actor_id, name_unified, name_variants,",
        "year_birth, year_death, year_pub_first_estc, year_pub_last_estc FROM estc_actors ea",
        'WHERE ea.actor_id = "' + str(actor_id) + '"'])
    cur.execute(query_str)
    res = list()
    for item in cur:
        res.append({
            'actor_id': item[0],
            'name_unified': item[1],
            'name_variants': item[2],
            'year_birth': item[3],
            'year_death': item[4],
            'year_pub_first_estc': item[5],
            'year_pub_last_estc': item[6]
        })
    if len (res) > 0:
        return res[0]
    else:
        return {
            'actor_id': None,
            'name_unified': None,
            'name_variants': None,
            'year_birth': None,
            'year_death': None,
            'year_pub_first_estc': None,
            'year_pub_last_estc': None
        }


ecco_id = '0145100106'  # hume vol 6
actor_id = '49226972'  # Hume



actor_id = '49226972'  # David Hume
# ecco_id = "0145000201"  # hume vol 1
ecco_id = '0145100106'  # hume vol 6
source_fragments = get_sources_and_indices_for_eccoid(ecco_id=ecco_id, cur=cur)
merged_source_fragments_per_ecco_id = get_combined_source_fragments_by_source_id(source_fragments)
merged_by_date = get_source_fragments_merged_by_date(merged_source_fragments_per_ecco_id, 0.5)
actor_coverage = get_filtered_source_coverage_for_id_per_actor(merged_by_date, ecco_id, actor_id, cur)
all_filtered_source_fragments = list()
for item in actor_coverage:
    for i in item['intervals']:
        all_filtered_source_fragments.append({'dst_start': i[0], 'dst_end': i[1]})
all_filtered_merged = get_source_coverage(all_filtered_source_fragments, get_text_for_id(ecco_id,cur))


def get_actor_coverage_results(actor_coverage_data, cur):
    actor_res = list()
    for item in actor_coverage_data:
        res_item = dict()
        for k in item.keys():
            if k != 'intervals':
                res_item[k] = item[k]
        actordata = get_data_for_actor_id(item['src_author_id'], cur)
        for k, v in actordata.items():
            if k != 'actor_id':
                res_item[k] = v
        actor_res.append(res_item)
    return actor_res


actor_id = '49226972'  # David Hume
ecco_vols = ["0145000201", "0145000202", "0145100103", "0145100104", "0145100105", "0145100106", "0145200108", "0145100107"]  # Hume vols


def get_vol_data(actor_id, ecco_vols, cur):
    results = dict()
    for ecco_id in ecco_vols:
        source_fragments = get_sources_and_indices_for_eccoid(ecco_id=ecco_id, cur=cur)
        merged_source_fragments_per_ecco_id = get_combined_source_fragments_by_source_id(source_fragments)
        merged_by_date = get_source_fragments_merged_by_date(merged_source_fragments_per_ecco_id, 0.5)
        actor_coverage = get_filtered_source_coverage_for_id_per_actor(merged_by_date, ecco_id, actor_id, cur)
        actor_coverage_res = get_actor_coverage_results(actor_coverage, cur)
        all_filtered_source_fragments = list()
        for item in actor_coverage:
            for i in item['intervals']:
                all_filtered_source_fragments.append({'dst_start': i[0], 'dst_end': i[1]})
        all_filtered_merged = get_source_coverage(all_filtered_source_fragments, get_text_for_id(ecco_id, cur))
        results[ecco_id] = {
            'actor_coverage': actor_coverage_res,
            'overall_coverage': all_filtered_merged
        }
    return results


hume_vol_data = get_vol_data(hume_actor_id, hume_vols, cur)

# for item in hume_vol_data:



# hume_coverage = get_source_coverage_for_multivol(hume_vols, hume_actor_id, cur)
write_json_results(hume_coverage, 'data/results/hume_coverage.json')


# todo: if an item is not merged into another item, cut it down to size so that it does not overlap with earlier
# items. After cutting, test for merge potential again.

# for item in hume_actor_coverage:


rider_frags = [f for f in source_fragments if f['src_author_id'] == '66343310']

# # testing code to check that grouping works correctly.
# grouped_len = 0
# intervals_n = 0
# all_grouped_intervals = []
# for k, v in ret_results.items():
#     intervals_n += len(v['intervals'])
#     all_grouped_intervals.extend(v['intervals'])
#     for item in v['intervals']:
#         grouped_len += item[1] - item[0]
# all_grouped_merged_again = merge_intervals(all_grouped_intervals)
# all_grouped_merged_again_total_length = 0
# for interval in all_grouped_merged_again:
#     all_grouped_merged_again_total_length += interval[1] - interval[0]


def get_filtered_source_coverage_for_id(ecco_id, actor_id, cur):
    source_fragments = get_sources_and_indices_for_eccoid(ecco_id=ecco_id, cur=cur)
    target_text = get_text_for_ecco_id(ecco_id, cur)[0]['text']
    filtered_fragments = filter_author_sources_and_indices(source_fragments, actor_id)
    source_coverage = get_source_coverage(filtered_fragments, target_text)
    return source_coverage



total_filtered = 0
for item in filtered_fragments:
    total_filtered += (item['dst_end'] - item['dst_start'])


with open('testout.txt', 'w') as outf:
    outf.writelines(list(set([row['src_id'] + '\n' for row in testres2])))


# Hume 1778 History edition in 8 volumes: estc_id T82472
# Rapin, T144219;
# Carte, T101742;
# Guthrie, T138171.

# 1. get all text reuse pieces for ecco_id
# Hume vol 1: 0145000201


thistext = get_text_for_ecco_id("0000100100", cur)


offset_data_path = "../data_input_text_reuse/data/work/tr_off/"
offset_data = get_offset_data(offset_data_path)



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

def get_tr_pairs_for_ecco_id_plain(ecco_id, cur):
    query_str = " ".join(
        ['WITH pieces AS (',
         'SELECT piece1_id as query_piece, piece2_id as other_piece FROM defrag_textreuses',
         'INNER JOIN defrag_pieces dp ON dp.piece_id  = piece1_id',
         'INNER JOIN textreuse_ids ti USING(trs_id)',
         'WHERE ti.manifestation_id = "' + ecco_id + '"',
         'UNION ALL',
         'SELECT piece2_id as query_piece, piece1_id as other_piece FROM defrag_textreuses',
         'INNER JOIN defrag_pieces dp ON dp.piece_id  = piece2_id',
         'INNER JOIN textreuse_ids ti USING(trs_id)',
         'WHERE ti.manifestation_id = "' + ecco_id + '"',
         ')',
         'SELECT ti_qp.manifestation_id AS qp_manifestation_id,dp2.trs_start AS qp_trs_start,',
         'dp2.trs_end AS qp_trs_end,mi.manifestation_id AS op_manifestation_id,',
         'mpd.publication_date AS op_publication_date, ai.actor_id AS op_author FROM pieces',
         'INNER JOIN defrag_pieces dp2 ON query_piece = dp2.piece_id',
         'INNER JOIN textreuse_ids ti_qp ON ti_qp.trs_id = dp2.trs_id',
         'INNER JOIN defrag_pieces dp3 ON other_piece = dp3.piece_id',
         'INNER JOIN textreuse_ids ti ON ti.trs_id = dp3.trs_id',
         'INNER JOIN manifestation_ids mi ON mi.manifestation_id = ti.manifestation_id',
         'INNER JOIN manifestation_publication_date mpd ON mi.manifestation_id_i = mpd.manifestation_id_i',
         'LEFT JOIN edition_mapping em ON em.manifestation_id_i = mi.manifestation_id_i',
         'LEFT JOIN edition_authors ea ON ea.edition_id_i = em.edition_id_i',
         'LEFT JOIN actor_ids ai ON ai.actor_id_i = ea.actor_id_i',
         'LEFT JOIN newspapers_core nc ON nc.article_id = ti.manifestation_id',
         'WHERE nc.article_id IS NULL'])
    cur.execute(query_str)
    res = list()
    for item in cur:
        res.append({
            'qp_manifestation_id': item[0],
            'qp_trs_start': item[1],
            'qp_trs_end': item[2],
            'op_manifestation_id': item[3],
            'op_publication_date': item[4],
            'op_author': item[5]
        })
    return res


def get_publication_date_for_manifestation_id(manifestation_id, cur):
    query_str = " ".join(
        ['SELECT publication_date from manifestation_publication_date mpd',
         'INNER JOIN manifestation_ids mi USING(manifestation_id_i)',
         'WHERE manifestation_id = "' + manifestation_id + '"'])
    cur.execute(query_str)
    res = list()
    for item in cur:
        res.append({
            'publication_date': item[0]
        })
    if len(res) > 0:
        return res[0]['publication_date']
    else:
        return None


# Testing "raw" reuses vs clustered


hume_actor_id = '49226972'  # David Hume
hume_ecco_id = hume_vols[1] # Charles I volume (5/6)
pubdate = get_publication_date_for_manifestation_id(hume_ecco_id, cur)
hume_raw_reuses = get_tr_pairs_for_ecco_id_plain(hume_ecco_id, cur)
hume_text = get_text_for_id(hume_ecco_id, cur)

# 1. filter out reuses later or as than query.
filtered = list()
for item in hume_raw_reuses:
    if item['op_publication_date'] is None:
        continue
    if item['op_publication_date'] < pubdate:
        filtered.append(item)


offset_map = dict()
for char_i in range(1, len(hume_text) + 1):  # The reuses start with 1-index and are inclusive
    # this_item = {'author_id': hume_actor_id, 'source_id': hume_ecco_id, 'pub_date': pubdate}
    this_item = None
    offset_map[char_i] = this_item

# 1. order entries by date, earliest to latest.
filtered = sorted(filtered, key=lambda d: d['op_publication_date'])

# 2. iterate filtered, add item if None in list at that index.
# Consider multiple items at same date? Each entry should be a list of dicts instead.
for item in filtered:
    for i in range(item['qp_trs_start'], item['qp_trs_end'] + 1):
        if offset_map[i] is None:
            if item['op_author'] is None:
                author_id = 'NA'
            else:
                author_id = item['op_author']
            offset_map[i] = [
                {'manifestation_id': item['op_manifestation_id'],
                 'publication_date': item['op_publication_date'],
                 'author': author_id}]
        elif offset_map[i] is not None:
            if offset_map[i][0]['publication_date'] == item['op_publication_date']:
                if item['op_author'] is None:
                    author_id = 'NA'
                else:
                    author_id = item['op_author']
                offset_map[i].append(
                    {'manifestation_id': item['op_manifestation_id'],
                     'publication_date': item['op_publication_date'],
                     'author': author_id})

# iterate the results to get totals:
actor_totals = dict()
overall_coverage = 0
for v in offset_map.values():
    if v is None:
        continue
    index_authors = [i['author'] for i in v]
    if hume_actor_id not in index_authors:
        overall_coverage += 1
        for item in v:
            if item['author'] in actor_totals.keys():
                actor_totals[item['author']] += 1
            else:
                actor_totals[item['author']] = 1
#
actor_coverage = list()
for k, v in actor_totals.items():
    actor_coverage.append({'src_author_id': k, 'combined_length': v, 'portion_covered': v/len(hume_text)})
actor_coverage = sorted(actor_coverage, key=lambda a: a['combined_length'], reverse=True)

# add actor data
for item in actor_coverage:
    actordata = get_data_for_actor_id(item['src_author_id'], cur)
    for k, v in actordata.items():
        item[k] = v


actordata_raw_reuses = pd.DataFrame(actor_coverage)
actordata_raw_reuses = actordata_raw_reuses[['combined_length', 'actor_id', 'name_unified']]

actordata_clustered = pd.DataFrame(hume_vol_data[hume_ecco_id]['actor_coverage'])
actordata_clustered = actordata_clustered[['combined_length', 'src_author_id', 'name_unified']]
actordata_clustered.rename(columns={'src_actor_id': 'actor_id'}, inplace=True)


rider_id = "66343310"
rider_totals = dict()
overall_coverage = 0
for v in offset_map.values():
    if v is None:
        continue
    index_authors = [i['author'] for i in v]
    if rider_id not in index_authors:
        continue
    else:
        for item in v:
            if item['author'] == rider_id:
                if item['manifestation_id'] not in rider_totals.keys():
                    rider_totals[item['manifestation_id']] = 1
                else:
                    rider_totals[item['manifestation_id']] += 1


grouped_fragments = list()
current_items = dict()
for char_i in tqdm(range(min(offset_map.keys()), max(offset_map.keys()) + 1)):
    # set temporary unique ids for items currently open:
    # current_ids = [item['manifestation_id'] + "-" + str(item['author']) for item in current_items]
    next_items = offset_map[char_i]
    if next_items is None:
        next_items = list()
    # check if next_items are in current items. If so, set their last index to current.
    for item in next_items:
        item_id = item['manifestation_id'] + "-" + str(item['author'])
        # If item is in current items, replace end index.
        if item_id in current_items.keys():
            current_items[item_id]['end'] = char_i
        # if item is not in current items, add a new entry to current.
        else:
            current_items[item_id] = {'manifestation_id': item['manifestation_id'],
                                      'author_id': item['author'],
                                      'start': char_i,
                                      'end': char_i}
    # after processing next items, check to see if current items need to be closed and added to finished items.
    next_items_keys = [item['manifestation_id'] + "-" + str(item['author']) for item in next_items]
    for key in list(current_items.keys()):
        if key not in next_items_keys:
            grouped_fragments.append(current_items.pop(key))
# Save the rest.
for key in list(current_items.keys()):
    grouped_fragments.append(current_items.pop(key))
