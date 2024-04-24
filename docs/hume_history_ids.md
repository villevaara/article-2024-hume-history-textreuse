SELECT edition_id, manifestation_id, actor_id, name_unified FROM edition_ids ei
INNER JOIN edition_mapping em ON em.edition_id_i = ei.edition_id_i
INNER JOIN manifestation_ids mi ON mi.manifestation_id_i = em.manifestation_id_i
LEFT JOIN edition_authors ea ON ea.edition_id_i = em.edition_id_i
LEFT JOIN actor_ids ai ON ai.actor_id_i = ea.actor_id_i
WHERE ei.edition_id = "T144219"


Hume, T82472; 

0145000202
0145100105
0145200108
0145100106
0145100103
0145100104
0145000201
0145100107

vol topic estc_id ecco_id year len has_headings notes 
I Caesar T82467 0162900301 1762 1207007 t
II Henry VII T82467 0162900302 1762 1307990 t
III Tudor T85928 0429000101 1759 1161167 t [same estc_id, 2 separate volumes]
IV Tudor T85928 0429000102 1759 1021296 t [same estc_id, 2 separate volumes]
V James I-Charles I T82483 0156400400 1754 1372119 t [in one volume]
VI Charles II-James II T82481 0162200200 1757 1364780 t

Rapin, T144219
actor_id = 59111095
1393700102
1394100115
1394000111
1393800105
1394000110
1394100113
1394100114
1393700103
1394000112
1393700101
1393800106
1393900109
1393800104
1393900108
1393900107

Carte, T101742
actor_id = 34794161
0018500104
0018400102
0018400101
0018500103

Guthrie, T138171
actor_id = 100194070
1679100104
1679000101
1679100103
