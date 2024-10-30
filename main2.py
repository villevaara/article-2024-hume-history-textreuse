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
import csv
from copy import copy


def get_shortened_header_text(text, max_length):
    if len(text) <= max_length:
        return text.replace('\n\n#', '')
    else:
        new_text = text.replace('\n\n#', '')[:max_length] + "[…]"
        return new_text


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
         'mpd.publication_date AS op_publication_date, ai.actor_id AS op_author,',
         'dp3.trs_start AS op_trs_start, dp3.trs_end AS op_trs_end FROM pieces',
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
            'op_trs_start': item[6],
            'op_trs_end': item[7],
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


def get_metadata_for_manifestation_id(manifestation_id, cur):
    query_str = " ".join(
        ['SELECT edition_id, manifestation_id, actor_id, name_unified, publication_date, title FROM edition_ids ei',
         'INNER JOIN edition_mapping em ON em.edition_id_i = ei.edition_id_i',
         'INNER JOIN manifestation_ids mi ON mi.manifestation_id_i = em.manifestation_id_i',
         'LEFT JOIN edition_authors ea ON ea.edition_id_i = em.edition_id_i',
         'LEFT JOIN actor_ids ai ON ai.actor_id_i = ea.actor_id_i',
         'LEFT JOIN manifestation_publication_date mpd ON mpd.manifestation_id_i = mi.manifestation_id_i',
         'LEFT JOIN manifestation_title mt ON mt.manifestation_id_i = mi.manifestation_id_i',
         'WHERE mi.manifestation_id = "' + str(manifestation_id) + '"'])
    cur.execute(query_str)
    res = list()
    for item in cur:
        res.append({
            'estc_id': item[0],
            'manifestation_id': item[1],
            'actor_id': item[2],
            'name_unified': item[3],
            'publication_date': item[4],
            'title': item[5],
        })
    if len(res) > 0:
        return res[0]
    else:
        return None


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


def get_page_count_for_id(text_id, cur):
    textq = ["SELECT ecco_pages FROM ecco_core ec",
             'WHERE ec.ecco_id = "' + text_id + '"']
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


def filter_by_date(raw_reuses, filter_id, keep, cur):
    pubdate = get_publication_date_for_manifestation_id(filter_id, cur)
    filtered = list()
    if keep == 'earlier':
        for item in raw_reuses:
            if item['op_publication_date'] is None:
                continue
            if item['op_publication_date'] < pubdate:
                filtered.append(item)
    elif keep == 'later':
        for item in raw_reuses:
            if item['op_publication_date'] is None:
                continue
            if item['op_publication_date'] > pubdate:
                filtered.append(item)
    return filtered


def get_offset_map(reuses, full_text):
    offset_map = dict()
    for char_i in range(1, len(full_text) + 1):  # The reuses start with 1-index and are inclusive
        # this_item = {'author_id': hume_actor_id, 'source_id': hume_ecco_id, 'pub_date': pubdate}
        this_item = None
        offset_map[char_i] = this_item
    # 1. order entries by date, earliest to latest.
    filtered = sorted(reuses, key=lambda d: d['op_publication_date'])
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
                    append_item = {
                        'manifestation_id': item['op_manifestation_id'],
                        'publication_date': item['op_publication_date'], 'author': author_id}
                    if append_item not in offset_map[i]:
                        offset_map[i].append(append_item)
    return offset_map


def get_overall_coverage(offset_map, filter_author_ids):
    overall_coverage = 0
    for v in offset_map.values():
        if v is None:
            continue
        index_authors = set([i['author'] for i in v])
        if len(index_authors.intersection(set(filter_author_ids))) == 0:
            overall_coverage += 1
    return overall_coverage


def get_actor_coverage(offset_map, filter_author_ids, text_length, cur):
    actor_totals = dict()
    for v in offset_map.values():
        if v is None:
            continue
        index_authors = set([i['author'] for i in v])
        # only add to totals if filter authors are not in the original authors
        if len(index_authors.intersection(set(filter_author_ids))) == 0:
            for author_id in index_authors:
                if author_id in actor_totals.keys():
                    actor_totals[author_id] += 1
                else:
                    actor_totals[author_id] = 1
    #
    actor_coverage = list()
    for k, v in actor_totals.items():
        actor_coverage.append({'src_author_id': k, 'combined_length': v, 'portion_covered': v/text_length})
    actor_coverage = sorted(actor_coverage, key=lambda a: a['combined_length'], reverse=True)
    # add actor data
    for item in actor_coverage:
        actordata = get_data_for_actor_id(item['src_author_id'], cur)
        for k, v in actordata.items():
            item[k] = v
    return actor_coverage


def get_fragments_grouped(offset_map):
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
    return grouped_fragments


def get_fragments_grouped_by_actor(offset_map, filter_actors=None):
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
            item_id = str(item['author'])
            # If item is in current items, replace end index.
            if item_id in current_items.keys():
                current_items[item_id]['end'] = char_i
            # if item is not in current items, add a new entry to current.
            else:
                current_items[item_id] = {'author_id': item['author'],
                                          'start': char_i,
                                          'end': char_i}
        # after processing next items, check to see if current items need to be closed and added to finished items.
        next_items_keys = [str(item['author']) for item in next_items]
        for key in list(current_items.keys()):
            if key not in next_items_keys:
                grouped_fragments.append(current_items.pop(key))
    # Save the rest.
    for key in list(current_items.keys()):
        grouped_fragments.append(current_items.pop(key))
    # if filter actors is set, filter
    if filter_actors is None:
        return grouped_fragments
    else:
        filtered_grouped = [f for f in grouped_fragments if f['author_id'] not in filter_actors]
        return filtered_grouped


def get_source_coverage_for_multivol(ecco_ids, actor_ids, cur):
    results = dict()
    total_target_len = 0
    total_source_len = 0
    for ecco_id in tqdm(ecco_ids):
        this_raw_reuses = get_tr_pairs_for_ecco_id_plain(ecco_id, cur)
        this_text = get_text_for_id(ecco_id, cur)
        this_earlier = filter_by_date(this_raw_reuses, ecco_id, 'earlier', cur)
        this_offset_map = get_offset_map(this_earlier, this_text)
        this_coverage = get_overall_coverage(this_offset_map, actor_ids)
        total_target_len += len(this_text)
        total_source_len += this_coverage
        results[ecco_id] = {'absolute': this_coverage, 'relative': this_coverage/len(this_text)}
    total_ratio = total_source_len / total_target_len
    results['total_ratio'] = total_ratio
    results['total_source_len'] = total_source_len
    results['total_target_len'] = total_target_len
    return results


def write_json_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


def add_actor_coverage_portion(actor_coverage):
    total_coverage = sum([i['combined_length'] for i in actor_coverage])
    for item in actor_coverage:
        item['portion_of_total_coverage'] = item['combined_length'] / total_coverage


def get_actors_for_volume(text_id, filter_actor_ids, cur):
    this_raw_reuses = get_tr_pairs_for_ecco_id_plain(text_id, cur)
    this_text = get_text_for_id(text_id, cur)
    this_earlier = filter_by_date(this_raw_reuses, text_id, 'earlier', cur)
    this_offset_map = get_offset_map(this_earlier, this_text)
    actor_coverage = get_actor_coverage(this_offset_map, filter_actor_ids, len(this_text), cur)
    add_actor_coverage_portion(actor_coverage)
    return actor_coverage


def read_actor_labels(author_labels_path):
    authdict = list()
    with open(author_labels_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            authdict.append({'name': row['name'], 'id': row['id'], 'faction': row['political_views']})
    filtered_list = [row for row in authdict if row['id'] != '']
    retdict = dict()
    for item in filtered_list:
        if item['faction'] in ['parliamentarian', 'whig']:
            item['broad_faction'] = 'whig'
        elif item['faction'] in ['jacobite', 'royalist', 'tory']:
            item['broad_faction'] = 'tory'
        else:
            item['broad_faction'] = 'other'
        retdict[item['id']] = {'name': item['name'], 'faction': item['faction'], 'broad_faction': item['broad_faction']}
    return retdict


def read_json_results(jsonfile):
    thisdata = json.load(open(jsonfile, 'r'))
    return thisdata


# todo: add login credentials in different file
cur = get_db_cursor()

rider_actor_id = "66343310" # William Rider needs to be filtered for separately, as his collection is erroneously dated
# before Hume's history's volumes I & II.
hume_actor_id = '49226972'  # David Hume
# the volumes are in order I-VI
hume_vols = ['0162900301', '0162900302', '0429000101', '0429000102', '0156400400', '0162200200']
# hume_coverage = get_source_coverage_for_multivol(hume_vols, [hume_actor_id, rider_actor_id], cur)
# write_json_results(hume_coverage, 'data/results/hume_coverage.json')
hume_coverage = read_json_results('data/results/hume_coverage.json')

rapin_actor_id = "59111095"
# The volumes are in order 1-15
rapin_vols = ['1393700101', '1393700102', '1393700103', '1393800104', '1393800105',
              '1393800106', '1393900107', '1393900108', '1393900109', '1394000110',
              '1394000111', '1394000112', '1394100113', '1394100114', '1394100115']
# rapin_coverage = get_source_coverage_for_multivol(rapin_vols, [rapin_actor_id], cur)
# write_json_results(rapin_coverage, 'data/results/rapin_coverage.json')
rapin_coverage = read_json_results('data/results/rapin_coverage.json')

carte_actor_id = "34794161"
carte_vols = ["0018400101", "0018400102", "0018500103", "0018500104"]
# carte_coverage = get_source_coverage_for_multivol(carte_vols, [carte_actor_id], cur)
# write_json_results(carte_coverage, 'data/results/carte_coverage.json')
carte_coverage = read_json_results('data/results/carte_coverage.json')

guthrie_actor_id = "100194070"
guthrie_vols = ["1679000101", "1679100103", "1679100104"]
# guthrie_coverage = get_source_coverage_for_multivol(guthrie_vols, [guthrie_actor_id], cur)
# write_json_results(guthrie_coverage, 'data/results/guthrie_coverage.json')
guthrie_coverage = read_json_results('data/results/guthrie_coverage.json')


volume_lengths = dict()
for vol_list in rapin_vols, hume_vols, carte_vols, guthrie_vols:
    for ecco_id in vol_list:
        volume_lengths[ecco_id] = len(get_text_for_id(ecco_id, cur))


# Overall reuse ratios plot (1).
plot1_data = pd.DataFrame([{'author': 'Hume', 'reuse ratio': hume_coverage['total_ratio']},
              {'author': 'Rapin', 'reuse ratio': rapin_coverage['total_ratio']},
              {'author': 'Carte', 'reuse ratio': carte_coverage['total_ratio']},
              {'author': 'Guthrie', 'reuse ratio': guthrie_coverage['total_ratio']}])
plot1_data.to_csv('plots/final/figure1_data.csv', index=False)

sns.barplot(plot1_data, x='author', y='reuse ratio')
plt.savefig('plots/plot1.png')
plt.close('all')

# Plots 2-5. Reuse ratios per volume.
plots2_to_5_data = pd.DataFrame([
    {'author': 'Hume', 'volume': '1', 'reuse ratio': hume_coverage['0162900301']['relative']},
    {'author': 'Hume', 'volume': '2', 'reuse ratio': hume_coverage['0162900302']['relative']},
    {'author': 'Hume', 'volume': '3', 'reuse ratio': hume_coverage['0429000101']['relative']},
    {'author': 'Hume', 'volume': '4', 'reuse ratio': hume_coverage['0429000102']['relative']},
    {'author': 'Hume', 'volume': '5', 'reuse ratio': hume_coverage['0156400400']['relative']},
    {'author': 'Hume', 'volume': '6', 'reuse ratio': hume_coverage['0162200200']['relative']},
    {'author': 'Rapin', 'volume': '1', 'reuse ratio': rapin_coverage['1393700101']['relative']},
    {'author': 'Rapin', 'volume': '2', 'reuse ratio': rapin_coverage['1393700102']['relative']},
    {'author': 'Rapin', 'volume': '3', 'reuse ratio': rapin_coverage['1393700103']['relative']},
    {'author': 'Rapin', 'volume': '4', 'reuse ratio': rapin_coverage['1393800104']['relative']},
    {'author': 'Rapin', 'volume': '5', 'reuse ratio': rapin_coverage['1393800105']['relative']},
    {'author': 'Rapin', 'volume': '6', 'reuse ratio': rapin_coverage['1393800106']['relative']},
    {'author': 'Rapin', 'volume': '7', 'reuse ratio': rapin_coverage['1393900107']['relative']},
    {'author': 'Rapin', 'volume': '8', 'reuse ratio': rapin_coverage['1393900108']['relative']},
    {'author': 'Rapin', 'volume': '9', 'reuse ratio': rapin_coverage['1393900109']['relative']},
    {'author': 'Rapin', 'volume': '10', 'reuse ratio': rapin_coverage['1394000110']['relative']},
    {'author': 'Rapin', 'volume': '11', 'reuse ratio': rapin_coverage['1394000111']['relative']},
    {'author': 'Rapin', 'volume': '12', 'reuse ratio': rapin_coverage['1394000112']['relative']},
    {'author': 'Rapin', 'volume': '13', 'reuse ratio': rapin_coverage['1394100113']['relative']},
    {'author': 'Rapin', 'volume': '14', 'reuse ratio': rapin_coverage['1394100114']['relative']},
    {'author': 'Rapin', 'volume': '15', 'reuse ratio': rapin_coverage['1394100115']['relative']},
    {'author': 'Carte', 'volume': '1', 'reuse ratio': carte_coverage['0018400101']['relative']},
    {'author': 'Carte', 'volume': '2', 'reuse ratio': carte_coverage['0018400102']['relative']},
    {'author': 'Carte', 'volume': '3', 'reuse ratio': carte_coverage['0018500103']['relative']},
    {'author': 'Carte', 'volume': '4', 'reuse ratio': carte_coverage['0018500104']['relative']},
    {'author': 'Guthrie', 'volume': '1', 'reuse ratio': guthrie_coverage['1679000101']['relative']},
    {'author': 'Guthrie', 'volume': '3', 'reuse ratio': guthrie_coverage['1679100103']['relative']},
    {'author': 'Guthrie', 'volume': '4', 'reuse ratio': guthrie_coverage['1679100104']['relative']},
])
plots2_to_5_data.to_csv('plots/final/figure2_data.csv', index=False)

f, axes = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(8,8))

sns.barplot(plots2_to_5_data[plots2_to_5_data['author'] == 'Hume'], x='volume', y='reuse ratio', ax=axes[0,0]).set(title='Hume', ylim=(0, 0.6), xlabel=None)
sns.barplot(plots2_to_5_data[plots2_to_5_data['author'] == 'Rapin'], x='volume', y='reuse ratio', ax=axes[0,1]).set(title='Rapin', ylim=(0, 0.6), xlabel=None)
sns.barplot(plots2_to_5_data[plots2_to_5_data['author'] == 'Guthrie'], x='volume', y='reuse ratio', ax=axes[1,0]).set(title='Guthrie', ylim=(0, 0.6))
sns.barplot(plots2_to_5_data[plots2_to_5_data['author'] == 'Carte'], x='volume', y='reuse ratio', ax=axes[1,1]).set(title='Carte', ylim=(0, 0.6))
plt.tight_layout()
plt.savefig('plots/plot_reuse_ratio_all.png')
plt.clf()
plt.close('all')


# for author in ['Hume', 'Rapin', 'Guthrie', 'Carte']:
#     sns.barplot(plots2_to_5_data[plots2_to_5_data['author'] == author], x='volume', y='reuse ratio').set(
#         title=author)
#     # plt.savefig('plots/plot_reuse_ratio_' + author + '.png')
#     # plt.close('all')
#

hume_actor_coverages = dict()
for hume_vol in tqdm(hume_vols):
    hume_actor_coverages[hume_vol] = (
        get_actors_for_volume(hume_vol, [hume_actor_id, rider_actor_id], cur))


actor_labels_path = "../data-public/authors-metadata/misc/author_metadata.csv"
actor_labels = read_actor_labels(actor_labels_path)


def get_actor_label(actor_id, actor_labels):
    if actor_id not in actor_labels:
        return 'other'
    elif actor_labels[actor_id]['faction'] == '':
        return 'other'
    else:
        return actor_labels[actor_id]['faction']


def get_actor_broad_label(actor_id, actor_labels):
    if actor_id not in actor_labels:
        return 'other'
    elif actor_labels[actor_id]['broad_faction'] == '':
        return 'other'
    else:
        return actor_labels[actor_id]['broad_faction']


def get_faction_coverage(actor_coverage, actor_labels, discount_anonymous=False):
    faction_coverage = {'whig': 0, 'tory': 0, 'other': 0}
    for item in actor_coverage:
        if discount_anonymous:
            if item['src_author_id'] is None or item['src_author_id'] == 'NA':
                continue
        if item['src_author_id'] in actor_labels.keys():
            faction_coverage[actor_labels[item['src_author_id']]['broad_faction']] += item['combined_length']
        else:
            faction_coverage['other'] += item['combined_length']
    return faction_coverage


hume_faction_coverages_by_vol = list()
volume_i = 1
for hume_vol in hume_vols:
    vol_length = volume_lengths[hume_vol]
    faction_coverage = get_faction_coverage(hume_actor_coverages[hume_vol], actor_labels)
    total_non_neutral = faction_coverage['whig'] + faction_coverage['tory']
    total_including_neutral = faction_coverage['whig'] + faction_coverage['tory'] + faction_coverage['other']
    rowdata = [{'volume': volume_i, 'faction': k, 'proportion': v / vol_length, 'absolute': v, 'proportion_factions': v / total_non_neutral, 'proportion_factions_including_neutral': v / total_including_neutral} for k, v in faction_coverage.items()]
    hume_faction_coverages_by_vol.extend(rowdata)
    volume_i += 1

# for reviewing data
alldata_plot6 = pd.DataFrame(hume_faction_coverages_by_vol)
alldata_plot6.to_csv('data/results/plot6_faction_coverage_by_volume.csv', index=False)

plotdata6 = pd.DataFrame(hume_faction_coverages_by_vol)
plotdata6 = plotdata6[plotdata6.faction != 'other']

# Average neutral
plotdata6_avg_neutral = pd.DataFrame(hume_faction_coverages_by_vol)
plotdata6_avg_neutral = plotdata6_avg_neutral[plotdata6_avg_neutral.faction == 'other']
plotdata6_avg_neutral['proportion_factions_including_neutral'].mean()

# plotdata6.plot(kind='bar', stacked=True, color=['blue', 'red'])

plotdata6.to_csv('plots/final/figure3_data.csv', index=False)

sns.histplot(
    data=plotdata6,
    x="volume", hue="faction",
    multiple="fill", stat="proportion", weights=plotdata6.proportion_factions,
    discrete=True, shrink=.8)
# sns.barplot(plotdata6, x='volume', y='proportion', hue='faction', dodge=False)
plt.savefig('plots/plot6.png')
plt.close('all')


# Check Rapin faction coverage by volume
rapin_actor_coverages = dict()
for rapin_vol in tqdm(rapin_vols):
    rapin_actor_coverages[rapin_vol] = (
        get_actors_for_volume(rapin_vol, [rapin_actor_id], cur))


rapin_faction_coverages_by_vol = list()
volume_i = 1
for rapin_vol in rapin_vols:
    vol_length = len(get_text_for_id(rapin_vol, cur))
    faction_coverage = get_faction_coverage(rapin_actor_coverages[rapin_vol], actor_labels)
    total_non_neutral = faction_coverage['whig'] + faction_coverage['tory']
    total_including_neutral = faction_coverage['whig'] + faction_coverage['tory'] + faction_coverage['other']
    rowdata = [{'volume': volume_i, 'faction': k, 'proportion': v / vol_length, 'absolute': v, 'proportion_factions': v / total_non_neutral, 'proportion_factions_including_neutral': v / total_including_neutral} for k, v in faction_coverage.items()]
    rapin_faction_coverages_by_vol.extend(rowdata)
    volume_i += 1

# for reviewing data
alldata_plot6_rapin = pd.DataFrame(rapin_faction_coverages_by_vol)
actor_rapin_vol12 = pd.DataFrame(rapin_actor_coverages[rapin_vols[11]])
# alldata_plot6_rapin.to_csv('data/results/plot6_faction_coverage_by_volume_rapin.csv', index=False)

plotdata6_rapin = pd.DataFrame(rapin_faction_coverages_by_vol)
plotdata6_rapin = plotdata6_rapin[plotdata6_rapin.faction != 'other']

# plotdata6.plot(kind='bar', stacked=True, color=['blue', 'red'])

sns.histplot(
    data=plotdata6_rapin,
    x="volume", hue="faction",
    multiple="fill", stat="proportion", weights=plotdata6_rapin.proportion_factions,
    discrete=True, shrink=.8)
# sns.barplot(plotdata6, x='volume', y='proportion', hue='faction', dodge=False)
plt.savefig('plots/plot6_rapin.png')
plt.close('all')


# figure 7
# summarized number of characters per actor in all Hume volumes together

length_meta = dict()
for k, v in volume_lengths.items():
    length_meta[k] = {'page_count': int(get_page_count_for_id(k, cur)), 'char_length': volume_lengths[k]}

hume_avg_char_per_page = (
    sum([length_meta[i]['char_length'] for i in hume_vols]) /
    sum([length_meta[i]['page_count'] for i in hume_vols]))


combined_hume_coverage_by_actor = dict()
for hac in hume_actor_coverages.values():
    for item in hac:
        if item['actor_id'] not in combined_hume_coverage_by_actor:
            combined_hume_coverage_by_actor[item['actor_id']] = item['combined_length']
        else:
            combined_hume_coverage_by_actor[item['actor_id']] += item['combined_length']

combined_hume_coverage_by_actor_list = [{'combined_length': v, 'actor_id': k} for k, v in combined_hume_coverage_by_actor.items()]
for item in combined_hume_coverage_by_actor_list:
    actordata = get_data_for_actor_id(item['actor_id'], cur)
    for k, v in actordata.items():
        item[k] = v
    item['faction'] = get_actor_label(item['actor_id'], actor_labels)
    item['pages_estimate'] = round(item['combined_length'] / hume_avg_char_per_page, 1)

actordata_raw_reuses = pd.DataFrame(combined_hume_coverage_by_actor_list)
actordata_raw_reuses = actordata_raw_reuses[['combined_length', 'pages_estimate', 'actor_id', 'name_unified', 'faction']]
actordata_raw_reuses['share'] = round(actordata_raw_reuses['combined_length'] / sum(actordata_raw_reuses['combined_length']) * 100, 1)
actordata_raw_reuses = actordata_raw_reuses.sort_values('combined_length', ascending=False)

plotdata7 = actordata_raw_reuses[actordata_raw_reuses['actor_id'].notnull()]
plotdata7 = plotdata7.sort_values('combined_length', ascending=False).head(20)

sns.barplot(plotdata7, x='name_unified', y='pages_estimate', hue='faction')

plt.savefig('plots/plot7.png')
plt.close('all')

tabledata_1 = actordata_raw_reuses[['actor_id', 'combined_length', 'pages_estimate', 'name_unified', 'faction', 'share']]
tabledata_1.to_csv('tables/tabledata_1.csv', index=False)


# Check reuses fragment by fragment.
def get_earliest_fragments_for_ecco_id(ecco_id, cur):
    vol_raw_reuses = get_tr_pairs_for_ecco_id_plain(ecco_id, cur)
    vol_text = get_text_for_id(ecco_id, cur)
    vol_earlier = filter_by_date(vol_raw_reuses, ecco_id, 'earlier', cur)
    vol_offset_map = get_offset_map(vol_earlier, vol_text)
    vol_fragments_grouped = get_fragments_grouped(vol_offset_map)
    for item in vol_fragments_grouped:
        item['fragment_length'] = item['end'] - item['start']
        item['target_id'] = ecco_id
        item_meta = get_metadata_for_manifestation_id(item['manifestation_id'], cur)
        for k, v in item_meta.items():
            item[k] = v
    return vol_fragments_grouped


def get_reuses_between_ecco_ids(ecco_id_qp, ecco_id_op, cur):
    vol_raw_reuses = get_tr_pairs_for_ecco_id_plain(ecco_id_qp, cur)
    qp_text = get_text_for_id(ecco_id_qp, cur)
    op_text = get_text_for_id(ecco_id_op, cur)
    qp_meta = get_metadata_for_manifestation_id(ecco_id_qp, cur)
    op_meta = get_metadata_for_manifestation_id(ecco_id_op, cur)
    op_meta['publication_date'] = op_meta['publication_date'].strftime('%d-%m-%Y')
    qp_meta['publication_date'] = qp_meta['publication_date'].strftime('%d-%m-%Y')
    filtered = list()
    for item in vol_raw_reuses:
        if item['op_manifestation_id'] != ecco_id_op:
            continue
        item['op_text'] = op_text[item['op_trs_start']:item['op_trs_end'] + 1]
        item['qp_text'] = qp_text[item['qp_trs_start']:item['qp_trs_end'] + 1]
        item['op_publication_date'] = item['op_publication_date'].strftime('%d-%m-%Y')
        filtered.append(item)
    return {'trs': filtered,
            'qp': qp_meta,
            'op': op_meta}


total_raw_reuses = 0
for vol in hume_vols:
    total_raw_reuses += len(get_tr_pairs_for_ecco_id_plain(vol, cur))


hume_fragments_grouped = list()
for vol in hume_vols:
    hume_fragments_grouped.extend(get_earliest_fragments_for_ecco_id(vol, cur))


fragments_df = pd.DataFrame(hume_fragments_grouped)
fragments_df.drop('actor_id', axis=1, inplace=True)
fragments_non_hume = fragments_df[fragments_df['author_id'] != '49226972']
fragments_na = fragments_df[fragments_df['author_id'] == 'NA']
fragments_na['manifestation_id'].value_counts()


hume_clarendon_reuses = get_reuses_between_ecco_ids(hume_vols[4], '0109700200', cur)
json.dump(hume_clarendon_reuses, open('hume_clarendon_reuses.json', 'w'), indent=4)

# Data for plots by header

def get_offset_data(offset_data_path):
    offset_data = dict()
    offset_files = glob(offset_data_path + "/tr_offset*.json")
    for file_path in offset_files:
        with open (file_path, 'r') as jsonf:
            offsets = json.load(jsonf)
            for doc_id, off_con in offsets.items():
                off_d = dict()
                for k, v in off_con.items():
                    off_d[int(k)] = v['header']
                offset_data[doc_id] = off_d
    return offset_data


offset_data_path = "../data_input_text_reuse/data/work/tr_off/"
offset_data = get_offset_data(offset_data_path)

cur = get_db_cursor()
# testtext = get_text_for_id(hume_vols[4], cur)
# test_headers = offset_data[hume_vols[4]]


def get_header_for_char_index(char_index, headers_data, clean_header_text=True):
    keylist = [k for k in headers_data.keys() if k <= char_index]
    if len(keylist) == 0:
        return {'header_index': 0, 'header_text': ""}
    header_index = max(keylist)
    header = headers_data[header_index]
    if clean_header_text:
        header = header.strip('\n ,#')
    return {'header_index': header_index, 'header_text': header}


def get_actor_data_per_header(ecco_id, offset_data, filter_actors, cur):
    vol_raw_reuses = get_tr_pairs_for_ecco_id_plain(ecco_id, cur)
    vol_text = get_text_for_id(ecco_id, cur)
    vol_earlier = filter_by_date(vol_raw_reuses, ecco_id, 'earlier', cur)
    vol_offset_map = get_offset_map(vol_earlier, vol_text)
    vol_fragments_grouped = get_fragments_grouped_by_actor(vol_offset_map, filter_actors=filter_actors)
    this_headers_data = copy(offset_data[ecco_id])
    this_headers_data[1] = ""
    header_fragments = dict()
    for k in this_headers_data.keys():
        header_fragments[k] = {'fragments': list(),
                               'header_text': this_headers_data[k].strip('\n ,#')}
    #
    # get fragments grouped by header:
    for fragment in vol_fragments_grouped:
        frag_header = get_header_for_char_index(char_index=fragment['start'], headers_data=this_headers_data)
        header_fragments[frag_header['header_index']]['fragments'].append(fragment)
    # get book part lengths
    #
    headers_i_ordered = list(this_headers_data.keys())
    headers_i_ordered.sort()
    for i in range(len(headers_i_ordered)):
        header_fragments[headers_i_ordered[i]]['part_start'] = headers_i_ordered[i]
        if i < len(headers_i_ordered) - 1:
            part_length = headers_i_ordered[i + 1] - headers_i_ordered[i]
            header_fragments[headers_i_ordered[i]]['part_end'] = headers_i_ordered[i + 1] - 1
        else:
            part_length = len(vol_text) - headers_i_ordered[i] + 1
            header_fragments[headers_i_ordered[i]]['part_end'] = len(vol_text)
        header_fragments[headers_i_ordered[i]]['part_length'] = part_length
        header_fragments[headers_i_ordered[i]]['part_share_of_book'] = part_length / len(vol_text)

    for k, v in header_fragments.items():
        author_coverage = dict()
        part_coverage_map = dict()
        total_author_coverage = 0
        for i in range(v['part_start'], v['part_end'] + 1):
            part_coverage_map[i] = 0
        for fragment in v['fragments']:
            fragment['length'] = fragment['end'] - fragment['start'] + 1
            total_author_coverage += fragment['length']
            if fragment['author_id'] in author_coverage.keys():
                author_coverage[fragment['author_id']]['characters'] += fragment['length']
            else:
                author_coverage[fragment['author_id']] = {'characters': fragment['length']}
            for i in range(fragment['start'], fragment['end'] + 1):
                part_coverage_map[i] = 1
        part_covered = sum(list(part_coverage_map.values()))
        header_fragments[k]['total_author_coverage'] = total_author_coverage
        header_fragments[k]['total_coverage'] = part_covered
        header_fragments[k]['total_coverage_proportional'] = part_covered / header_fragments[k]['part_length']
        # add proportional coverage data to author coverage
        for author_data in author_coverage.values():
            author_data['portion_of_total_author_coverage'] = author_data['characters'] / total_author_coverage
            author_data['portion_of_total_coverage'] = author_data['characters'] / part_covered
        header_fragments[k]['author_coverage'] = author_coverage
    return header_fragments


def get_political_faction_summaries_to_header_data(vol_header_data, actor_labels):
    retdata = list()
    for v in vol_header_data.values():
        sums = {'whig': 0, 'tory': 0, 'other': 0}
        for fragment in v['fragments']:
            if fragment['author_id'] in actor_labels.keys():
                actor_faction = actor_labels[fragment['author_id']]['broad_faction']
            else:
                actor_faction = 'other'
            sums[actor_faction] += fragment['length']
        all_factions_sum = sum(sums.values())
        faction_factions_sum = sums['tory'] + sums['whig']
        for key, val in sums.items():
            retdict = dict()
            keepkeys = ['header_text', 'part_start', 'part_length']
            for k in v.keys():
                if k in keepkeys:
                    retdict[k] = v[k]
            retdict['faction'] = key
            retdict['faction_characters'] = val
            if all_factions_sum > 0:
                retdict['share_including_other'] = val / all_factions_sum
            else:
                retdict['share_including_other'] = 0
            if faction_factions_sum > 0:
                retdict['share_excluding_other'] = val / faction_factions_sum
            else:
                retdict['share_excluding_other'] = 0
            retdict['percentage_of_part_length'] = val / v['part_length']
            retdata.append(retdict)
    return retdata


cur = get_db_cursor()
hume_actor_header_data_vol5 = get_actor_data_per_header(hume_vols[4], offset_data, [hume_actor_id, rider_actor_id], cur)
hume_actor_header_data_vol6 = get_actor_data_per_header(hume_vols[5], offset_data, [hume_actor_id, rider_actor_id], cur)


charles_i_actor_id = '67750325'


def get_vol_header_overview_df(vol_header_data):
    keep_keys = ['header_text', 'part_start', 'part_end', 'part_length', 'part_share_of_book',
                 'total_author_coverage', 'total_coverage', 'total_coverage_proportional']
    overview_data = list(vol_header_data.values())
    overview_data_filtered = list()
    for item in overview_data:
        new_item = dict()
        for k, v in item.items():
            if k in keep_keys:
                new_item[k] = v
        overview_data_filtered.append(new_item)
    overview_data_filtered = sorted(overview_data_filtered, key=lambda k: k['part_start'])
    vol_header_data_df = pd.DataFrame(overview_data_filtered)
    return vol_header_data_df


vol5_parts_df = get_vol_header_overview_df(hume_actor_header_data_vol5)
vol5_parts_df = vol5_parts_df.iloc[3:19]
vol5_parts_df['volume'] = 5
vol6_parts_df = get_vol_header_overview_df(hume_actor_header_data_vol6)
vol6_parts_df = vol6_parts_df.iloc[3:15]
vol6_parts_df['volume'] = 6
hume_vols_df = pd.concat([vol5_parts_df, vol6_parts_df], ignore_index=True)
hume_vols_df.reset_index(inplace=True)
hume_vols_df['header_text_short'] = hume_vols_df['header_text'].str.split('\\n\\n# ').str[-1].str.split('THE HISTORY OF GREAT BRITAIN. ').str[-1].str[:50] + " […]"
hume_vols_df.to_csv('plots/final/figure4_data.csv', index=False)

f, ax = plt.subplots(figsize=(5, 14))
sns.barplot(hume_vols_df, x='total_coverage_proportional',
            y='header_text_short', orient='h', hue='volume')
plt.savefig('plots/hume_vols_overview.png', bbox_inches='tight')
plt.clf()
plt.close('all')


hume_actor_header_data_vol5_factions_df = pd.DataFrame(get_political_faction_summaries_to_header_data(hume_actor_header_data_vol5, actor_labels))
hume_actor_header_data_vol5_factions_df['volume'] = 5
hume_actor_header_data_vol5_factions_df = hume_actor_header_data_vol5_factions_df.iloc[6:54]
hume_actor_header_data_vol6_factions_df = pd.DataFrame(get_political_faction_summaries_to_header_data(hume_actor_header_data_vol6, actor_labels))
hume_actor_header_data_vol6_factions_df['volume'] = 6
hume_actor_header_data_vol6_factions_df = hume_actor_header_data_vol6_factions_df.iloc[6:42]
hume_actor_header_data_vols_factions_df = pd.concat([hume_actor_header_data_vol5_factions_df, hume_actor_header_data_vol6_factions_df], ignore_index=True)
hume_actor_header_data_vols_factions_df['header_text_short'] = hume_actor_header_data_vols_factions_df['header_text'].str.split('\\n\\n# ').str[-1].str.split('THE HISTORY OF GREAT BRITAIN. ').str[-1].str[:50] + " […]"
hume_actor_header_data_vols_factions_df_plotdata = hume_actor_header_data_vols_factions_df[hume_actor_header_data_vols_factions_df['faction'] != 'other']
hume_actor_header_data_vols_factions_df_plotdata.to_csv('plots/final/figure5_data.csv', index=False)

f, ax = plt.subplots(figsize=(5, 14))
sns.histplot(data=hume_actor_header_data_vols_factions_df_plotdata,
             y='header_text_short', hue='faction',
             multiple='fill', stat="proportion", discrete=True, shrink=.8,
             weights=hume_actor_header_data_vols_factions_df_plotdata.share_excluding_other,)
plt.savefig('plots/tories_vs_whigs_hume5-6_proportional_stacked.png', bbox_inches='tight')
plt.clf()
plt.close('all')

f, ax = plt.subplots(figsize=(5, 14))
sns.barplot(hume_actor_header_data_vols_factions_df_plotdata, x='percentage_of_part_length',
            y='header_text_short', orient='h', hue='faction')
plt.savefig('plots/tories_vs_whigs_hume5-6_proportional_side_by_side.png', bbox_inches='tight')
plt.clf()
plt.close('all')

hume_actor_header_data_vol5_ec = get_actor_data_per_header(hume_vols[4], offset_data, [hume_actor_id, rider_actor_id, charles_i_actor_id], cur)
hume_actor_header_data_vol6_ec = get_actor_data_per_header(hume_vols[5], offset_data, [hume_actor_id, rider_actor_id, charles_i_actor_id], cur)
hume_actor_header_data_vol5_factions_df_ec = pd.DataFrame(get_political_faction_summaries_to_header_data(hume_actor_header_data_vol5_ec, actor_labels))
hume_actor_header_data_vol5_factions_df_ec['volume'] = 5
hume_actor_header_data_vol5_factions_df_ec = hume_actor_header_data_vol5_factions_df_ec.iloc[6:54]
hume_actor_header_data_vol6_factions_df_ec = pd.DataFrame(get_political_faction_summaries_to_header_data(hume_actor_header_data_vol6_ec, actor_labels))
hume_actor_header_data_vol6_factions_df_ec['volume'] = 6
hume_actor_header_data_vol6_factions_df_ec = hume_actor_header_data_vol6_factions_df_ec.iloc[6:42]
hume_actor_header_data_vols_factions_df_ec = pd.concat([hume_actor_header_data_vol5_factions_df_ec, hume_actor_header_data_vol6_factions_df_ec], ignore_index=True)
hume_actor_header_data_vols_factions_df_ec['header_text_short'] = hume_actor_header_data_vols_factions_df_ec['header_text'].str.split('\\n\\n# ').str[-1].str.split('THE HISTORY OF GREAT BRITAIN. ').str[-1].str[:50] + " […]"
hume_actor_header_data_vols_factions_df_plotdata_ec = hume_actor_header_data_vols_factions_df_ec[hume_actor_header_data_vols_factions_df_ec['faction'] != 'other']

f, ax = plt.subplots(figsize=(5, 14))
sns.histplot(data=hume_actor_header_data_vols_factions_df_plotdata,
             y='header_text_short', hue='faction',
             multiple='fill', stat="proportion", discrete=True, shrink=.8,
             weights=hume_actor_header_data_vols_factions_df_plotdata.share_excluding_other,)
plt.savefig('plots/tories_vs_whigs_hume5-6_proportional_stacked_ec.png', bbox_inches='tight')
plt.clf()
plt.close('all')

f, ax = plt.subplots(figsize=(5, 14))
sns.barplot(hume_actor_header_data_vols_factions_df_plotdata, x='percentage_of_part_length',
            y='header_text_short', orient='h', hue='faction')
plt.savefig('plots/tories_vs_whigs_hume5-6_proportional_side_by_side_ec.png', bbox_inches='tight')
plt.clf()
plt.close('all')


# Text reuse for Rapin
cur = get_db_cursor()
rapin_actor_header_data_vol11 = get_actor_data_per_header(rapin_vols[10], offset_data, [rapin_actor_id], cur)
rapin_actor_header_data_vol12 = get_actor_data_per_header(rapin_vols[11], offset_data, [rapin_actor_id], cur)
rapin_vol11_factions_df = pd.DataFrame(get_political_faction_summaries_to_header_data(rapin_actor_header_data_vol11, actor_labels))
rapin_vol11_factions_df['volume'] = 11
rapin_vol11_factions_df = rapin_vol11_factions_df.iloc[6:69]
rapin_vol12_factions_df = pd.DataFrame(get_political_faction_summaries_to_header_data(rapin_actor_header_data_vol12, actor_labels))
rapin_vol12_factions_df['volume'] = 12
rapin_vol12_factions_df = rapin_vol12_factions_df.iloc[6:36]
rapin_vols_factions_df = pd.concat([rapin_vol11_factions_df, rapin_vol12_factions_df], ignore_index=True)
rapin_vols_factions_df['header_text_short'] = rapin_vols_factions_df['header_text'].apply(get_shortened_header_text, max_length=50)
rapin_vols_factions_df_plotdata = rapin_vols_factions_df[rapin_vols_factions_df['faction'] != 'other']
rapin_vols_factions_df_plotdata.to_csv('plots/final/figure6_data.csv', index=False)

f, ax = plt.subplots(figsize=(5, 14))
sns.barplot(rapin_vols_factions_df_plotdata, x='percentage_of_part_length',
            y='header_text_short', orient='h', hue='faction')
plt.savefig('plots/tories_vs_whigs_rapin11-12_proportional_side_by_side.png', bbox_inches='tight')
plt.clf()
plt.close('all')

f, ax = plt.subplots(figsize=(5, 14))
sns.histplot(data=rapin_vols_factions_df_plotdata,
             y='header_text_short', hue='faction',
             multiple='fill', stat="proportion", discrete=True, shrink=.8,
             weights=rapin_vols_factions_df_plotdata.share_excluding_other,)
plt.savefig('plots/tories_vs_whigs_rapin11-12_proportional_stacked.png', bbox_inches='tight')
plt.clf()
plt.close('all')


# text reuse by chapter for Rapin, excluding Charles I
cur = get_db_cursor()
rapin_actor_header_data_vol11_exci = get_actor_data_per_header(rapin_vols[10], offset_data, [rapin_actor_id, charles_i_actor_id], cur)
rapin_actor_header_data_vol12_exci = get_actor_data_per_header(rapin_vols[11], offset_data, [rapin_actor_id, charles_i_actor_id], cur)
rapin_vol11_factions_df_exci = pd.DataFrame(get_political_faction_summaries_to_header_data(rapin_actor_header_data_vol11_exci, actor_labels))
rapin_vol11_factions_df_exci['volume'] = 11
rapin_vol11_factions_df_exci = rapin_vol11_factions_df_exci.iloc[6:69]
rapin_vol12_factions_df_exci = pd.DataFrame(get_political_faction_summaries_to_header_data(rapin_actor_header_data_vol12_exci, actor_labels))
rapin_vol12_factions_df_exci['volume'] = 12
rapin_vol12_factions_df_exci = rapin_vol12_factions_df_exci.iloc[6:36]
rapin_vols_factions_df_exci = pd.concat([rapin_vol11_factions_df_exci, rapin_vol12_factions_df_exci], ignore_index=True)
rapin_vols_factions_df_exci['header_text_short'] = rapin_vols_factions_df_exci['header_text'].apply(get_shortened_header_text, max_length=50)
rapin_vols_factions_df_plotdata_exci = rapin_vols_factions_df_exci[rapin_vols_factions_df_exci['faction'] != 'other']

f, ax = plt.subplots(figsize=(5, 14))
sns.barplot(rapin_vols_factions_df_plotdata_exci, x='percentage_of_part_length',
            y='header_text_short', orient='h', hue='faction')
plt.savefig('plots/tories_vs_whigs_rapin11-12_proportional_side_by_side_exci.png', bbox_inches='tight')
plt.clf()
plt.close('all')

f, ax = plt.subplots(figsize=(5, 14))
sns.histplot(data=rapin_vols_factions_df_plotdata_exci,
             y='header_text_short', hue='faction',
             multiple='fill', stat="proportion", discrete=True, shrink=.8,
             weights=rapin_vols_factions_df_plotdata.share_excluding_other,)
plt.savefig('plots/tories_vs_whigs_rapin11-12_proportional_stacked_exci.png', bbox_inches='tight')
plt.clf()
plt.close('all')


cur = get_db_cursor()
guth_actor_header_data_vol4 = get_actor_data_per_header(guthrie_vols[2], offset_data, [guthrie_actor_id], cur)
guth_factions_vol4_df = pd.DataFrame(get_political_faction_summaries_to_header_data(guth_actor_header_data_vol4, actor_labels))
guth_factions_vol4_df = guth_factions_vol4_df.iloc[0:15]
guth_factions_vol4_df['header_text_short'] = guth_factions_vol4_df['header_text'].apply(get_shortened_header_text, max_length=50)
guth_factions_vol4_df_plotdata = guth_factions_vol4_df[guth_factions_vol4_df['faction'] != 'other']
guth_factions_vol4_df_plotdata.to_csv('plots/final/figure7_data.csv', index=False)

f, ax = plt.subplots(figsize=(5, 4))
sns.barplot(guth_factions_vol4_df_plotdata, x='percentage_of_part_length',
            y='header_text_short', orient='h', hue='faction')
plt.savefig('plots/tories_vs_whigs_guthrie4_proportional_side_by_side.png', bbox_inches='tight')
plt.clf()
plt.close('all')

f, ax = plt.subplots(figsize=(5, 4))
sns.histplot(data=guth_factions_vol4_df_plotdata,
             y='header_text_short', hue='faction',
             multiple='fill', stat="proportion", discrete=True, shrink=.8,
             weights=guth_factions_vol4_df_plotdata.share_excluding_other,)
plt.savefig('plots/tories_vs_whigs_guthrie4_proportional_stacked.png', bbox_inches='tight')
plt.clf()
plt.close('all')


cur = get_db_cursor()
carte_actor_header_data_vol4 = get_actor_data_per_header(carte_vols[-1], offset_data, [carte_actor_id], cur)
carte_factions_vol4_df = pd.DataFrame(get_political_faction_summaries_to_header_data(carte_actor_header_data_vol4, actor_labels))
carte_factions_vol4_df = carte_factions_vol4_df.iloc[0:12]
carte_factions_vol4_df['header_text_short'] = carte_factions_vol4_df['header_text'].apply(get_shortened_header_text, max_length=50)
carte_factions_vol4_df_plotdata = carte_factions_vol4_df[guth_factions_vol4_df['faction'] != 'other']
carte_factions_vol4_df_plotdata.to_csv('plots/final/figure8_data.csv', index=False)

f, ax = plt.subplots(figsize=(5, 4))
sns.barplot(carte_factions_vol4_df_plotdata, x='percentage_of_part_length',
            y='header_text_short', orient='h', hue='faction')
plt.savefig('plots/tories_vs_whigs_carte4_proportional_side_by_side.png', bbox_inches='tight')
plt.clf()
plt.close('all')

f, ax = plt.subplots(figsize=(5, 4))
sns.histplot(data=carte_factions_vol4_df_plotdata,
             y='header_text_short', hue='faction',
             multiple='fill', stat="proportion", discrete=True, shrink=.8,
             weights=carte_factions_vol4_df_plotdata.share_excluding_other,)
plt.savefig('plots/tories_vs_whigs_carte4_proportional_stacked.png', bbox_inches='tight')
plt.clf()
plt.close('all')




rapin_vol11_parts_df = get_vol_header_overview_df(rapin_actor_header_data_vol11)
rapin_vol11_parts_df = rapin_vol11_parts_df.iloc[2:24]
rapin_vol11_parts_df['volume'] = 11
rapin_vol12_parts_df = get_vol_header_overview_df(rapin_actor_header_data_vol12)
rapin_vol12_parts_df = rapin_vol12_parts_df.iloc[2:13]
rapin_vol12_parts_df['volume'] = 12
rapin_vols_parts_df = pd.concat([rapin_vol11_parts_df, rapin_vol12_parts_df], ignore_index=True)
rapin_vols_parts_df.reset_index(inplace=True)
rapin_vols_parts_df['header_text_short'] = rapin_vols_parts_df['header_text'].apply(get_shortened_header_text, max_length=50)


f, ax = plt.subplots(figsize=(5, 14))
sns.barplot(rapin_vols_parts_df, x='total_coverage_proportional',
            y='header_text_short', orient='h', hue='volume')
plt.savefig('plots/rapin_vols_overview.png', bbox_inches='tight')
plt.clf()
plt.close('all')


# Checking data ...
def get_volume_actor_coverages_df_with_labels(volume_actor_coverage_list, actor_labels):
    actors_this_vol = pd.DataFrame(volume_actor_coverage_list)
    actor_labels_l = list()
    for k, v in actor_labels.items():
        v['src_author_id'] = k
        actor_labels_l.append(v)
    actor_labels_df = pd.DataFrame(actor_labels_l)
    actor_labels_df = actor_labels_df[['broad_faction', 'src_author_id']]
    actors_this_vol = pd.merge(actors_this_vol, actor_labels_df, on='src_author_id', how='left')
    return actors_this_vol


rap_lab12 = get_volume_actor_coverages_df_with_labels(rapin_actor_coverages[rapin_vols[11]], actor_labels)
rap_lab11 = get_volume_actor_coverages_df_with_labels(rapin_actor_coverages[rapin_vols[10]], actor_labels)
rap_frag12 = pd.DataFrame(get_earliest_fragments_for_ecco_id(rapin_vols[11], cur))
rap_frag11 = pd.DataFrame(get_earliest_fragments_for_ecco_id(rapin_vols[10], cur))
rap_frag11_c = rap_frag11[rap_frag11['author_id'] == charles_i_actor_id]

sum(rap_lab12[(rap_lab12['broad_faction'] == 'tory') & (rap_lab12['actor_id'] != '67750325')]['portion_of_total_coverage'])
sum(rap_lab12[(rap_lab12['broad_faction'] == 'tory')]['portion_of_total_coverage'])
sum(rap_lab12[(rap_lab12['broad_faction'] == 'whig')]['portion_of_total_coverage'])


hum_lab6 = get_volume_actor_coverages_df_with_labels(hume_actor_coverages[hume_vols[5]], actor_labels)
hum_lab5 = get_volume_actor_coverages_df_with_labels(hume_actor_coverages[hume_vols[4]], actor_labels)

car_lab4 = get_volume_actor_coverages_df_with_labels(get_actors_for_volume(carte_vols[3], [carte_actor_id], cur), actor_labels)
gut_lab4 = get_volume_actor_coverages_df_with_labels(get_actors_for_volume(guthrie_vols[2], [guthrie_actor_id], cur), actor_labels)
gut_frag4 = pd.DataFrame(get_earliest_fragments_for_ecco_id(guthrie_vols[2], cur))
gut_frag4['header_text'] = gut_frag4['start'].apply(get_header_for_char_index, headers_data=offset_data[guthrie_vols[2]])
gut_frag4[['F', 'header_text']] = gut_frag4.header_text.apply(pd.Series)
gut_frag4.drop('F', axis=1, inplace=True)
gut_frag4_b3 = gut_frag4[gut_frag4['header_text'] == 'BOOK III.']
temp_sums = gut_frag4_b3.groupby(['author_id'])['fragment_length'].sum()

hum_frag5 = pd.DataFrame(get_earliest_fragments_for_ecco_id(hume_vols[4], cur))
hum_frag5['header_text'] = hum_frag5['start'].apply(get_header_for_char_index, headers_data=offset_data[hume_vols[4]])
hum_frag5[['header_index', 'header_text']] = hum_frag5.header_text.apply(pd.Series)
hum_frag5_x = hum_frag5[hum_frag5['header_text'] == 'CHAP. X. Mutiny of the army.-The King seized by Joyce.-The army march against the parliament.-The army subdue the parliament.-The King flies to the isle of Wight.-Second civil war.-Invasion from Scotl ...']
temp_sums = hum_frag5_x.groupby(['author_id'])['fragment_length'].sum()

# Get top X authors for Hume volume 5, and their reuses by chapter.
top_x_authors = 10
hum_frag5_sums_by_author = pd.DataFrame(hum_frag5.groupby(['author_id'])['fragment_length'].sum())
hum_frag5_top_authors = hum_frag5_sums_by_author.sort_values(by=['fragment_length'], ascending=False).iloc[2:(top_x_authors + 2)]
hum_frag5_top_authors.reset_index(inplace=True)

hum_actordata = list()
for actor in list(hum_frag5_top_authors['author_id']):
    res = get_data_for_actor_id(actor, cur)
    label = get_actor_label(actor_id=actor, actor_labels=actor_labels)
    hum_actordata.append({'author_id': res['actor_id'], 'author_name': res['name_unified'], 'author_label': label})
hum_actordata_df = pd.DataFrame(hum_actordata)
hum_frag5_top_authors = pd.merge(hum_frag5_top_authors, hum_actordata_df, on='author_id')

hum_frag5_auth_by_chapter = pd.DataFrame(hum_frag5.groupby(['author_id', 'header_index', 'header_text'])['fragment_length'].sum())
hum_frag5_auth_by_chapter.reset_index(inplace=True)
hum_frag5_auth_by_chapter['header_index'] = hum_frag5_auth_by_chapter['header_index'].astype(int)
hum_frag5_auth_by_chapter_top = pd.merge(hum_frag5_top_authors, hum_frag5_auth_by_chapter, how='left', on='author_id')
hum_frag5_auth_by_chapter_top['fragment_length'] = hum_frag5_auth_by_chapter_top['fragment_length_y']
hum_frag5_auth_by_chapter_top.drop(['fragment_length_x', 'fragment_length_y'], axis=1, inplace=True)
hum_frag5_auth_by_chapter_rest = hum_frag5_auth_by_chapter[~hum_frag5_auth_by_chapter['author_id'].isin(hum_frag5_top_authors['author_id'])]
hum_frag5_auth_by_chapter_rest = pd.DataFrame(hum_frag5_auth_by_chapter_rest.groupby(['header_index','header_text'])['fragment_length'].sum())
hum_frag5_auth_by_chapter_rest.reset_index(inplace=True)
hum_frag5_auth_by_chapter_rest['author_id'] = "NA"
hum_frag5_auth_by_chapter_rest['author_name'] = "Others"
hum_frag5_auth_by_chapter_rest['author_label'] = "varies"
hum_frag5_auth_by_chapter_top_merged = pd.concat([hum_frag5_auth_by_chapter_top, hum_frag5_auth_by_chapter_rest], ignore_index=True)
hum_frag5_auth_by_chapter_top_merged['pages'] = hum_frag5_auth_by_chapter_top_merged['fragment_length'] / hume_avg_char_per_page
hum_frag5_auth_by_chapter_top_merged['header_text_short'] = hum_frag5_auth_by_chapter_top_merged['header_text'].apply(get_shortened_header_text, max_length=50)
hum_frag5_auth_by_chapter_top_merged.sort_values(by='header_index', inplace=True)
hum_frag5_auth_by_chapter_top_merged.to_csv('plots/final/figure9_data.csv', index=False)

f, ax = plt.subplots(figsize=(10, 10))
hue_order = list(hum_frag5_top_authors['author_name'])
hue_order.sort()
hue_order.insert(0, 'Others')
thisplot = sns.histplot(data=hum_frag5_auth_by_chapter_top_merged,
             y='header_text_short', hue='author_name',
             multiple='stack', discrete=True, shrink=.8,
             weights=hum_frag5_auth_by_chapter_top_merged.pages,
             palette=sns.color_palette("Paired"),
             hue_order=hue_order)
ax.set_xlabel("Pages")

hatches = ['-', '+', '/', '\\', 'x', '*', 'o', 'O', '.', 'OO', '++', '--']
for i,thisbar in enumerate(thisplot.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[i])

plt.savefig('plots/hume_vol5_top10_authors_stacked.png', bbox_inches='tight')
plt.clf()
plt.close('all')

# Get number of authors per chapter for Hume vol 5
hum_frag5['author_label'] = hum_frag5['author_id'].apply(get_actor_broad_label, actor_labels=actor_labels)
hum5_chap_nunique_authors = pd.DataFrame(hum_frag5.groupby(['header_index', 'header_text'])['author_id'].nunique())
hum5_chap_nunique_authors.reset_index(inplace=True)
hum5_chap_nunique_authors['header_text_short'] = hum5_chap_nunique_authors['header_text'].apply(get_shortened_header_text, max_length=50)
hum5_chap_nunique_authors = hum5_chap_nunique_authors.iloc[0:16]  # Drop advertisements chapter
hum5_chap_nunique_authors[['author_id', 'header_text_short']].to_csv('tables/table_2.csv', index=False)

hum5_chap_label_authors = pd.DataFrame(hum_frag5.groupby(['header_index', 'header_text', 'author_label'])['author_id'].nunique().unstack())
hum5_chap_label_authors.reset_index(inplace=True)
hum5_chap_label_authors['header_text'] = hum5_chap_label_authors['header_text'].apply(get_shortened_header_text, max_length=50)
hum5_chap_label_authors = hum5_chap_label_authors.iloc[0:16]  # Drop advertisements chapter
hum5_chap_label_authors = hum5_chap_label_authors[['header_text', 'other', 'tory', 'whig']]
hum5_chap_label_authors['other'] = hum5_chap_label_authors['other'].astype(int)
hum5_chap_label_authors['whig'] = hum5_chap_label_authors['whig'].astype(int)
hum5_chap_label_authors.loc[hum5_chap_label_authors['tory'].isna(), 'tory'] = 0
hum5_chap_label_authors['tory'] = hum5_chap_label_authors['tory'].astype(int)
hum5_chap_label_authors.to_csv('tables/table_2_2.csv', index=False)


# number of characters for each author under this header
# proportional coverage per author header
# length of text under header
# length of text not under other authors (in other words: original text)
# header_author_sums = dict()
