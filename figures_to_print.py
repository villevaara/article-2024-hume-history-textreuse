import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# import itertools
import matplotlib as mpl
# from matplotlib.pyplot import title
from  matplotlib.ticker import PercentFormatter
from PIL import Image


def get_shortened_header_text(text, max_length):
    text = text.split('\\n\\n# ')[-1].split('THE HISTORY OF GREAT BRITAIN.')[-1]
    text = " ".join(text.replace('\n\n#', '').replace('.', '. ').split())
    text = text.replace('parlilament', 'parliament')
    if len(text) <= max_length:
        return text
    else:
        nt_split = text.split()
        nt_limited = ""
        for i in range(0, len(nt_split)):
            nt_limited += nt_split[i] + " "
            if len(nt_limited.strip()) > max_length:
                nt_limited = " ".join(nt_split[0:i])
                nt_limited = nt_limited.strip(' -')
                break
        nt_limited = nt_limited + " […]"
        return nt_limited


def save_image_in_all_formats(thisplot, figname, tight=False):
    if tight:
        thisplot.savefig('plots/final/' + figname + '.png', bbox_inches='tight')
        thisplot.savefig('plots/final/' + figname + '.eps', format='eps', bbox_inches='tight')
    else:
        thisplot.savefig('plots/final/' + figname + '.png')
        thisplot.savefig('plots/final/' + figname + '.eps', format='eps')
    Image.open('plots/final/' + figname + '.png').convert('RGB').save(
        'plots/final/' + figname + '.jpg', 'JPEG', quality=95)
    thisplot.clf()
    thisplot.close('all')


# Defaults

sns.set_theme(style="white",
              palette=['grey', 'lightgrey', 'black', 'white', 'darkgrey'],
              rc= {"patch.edgecolor": "black",
                   "figure.dpi": 400})


# Figure 1.

figure1_data = pd.read_csv('plots/final/figure1_data.csv')
figure1_data.sort_values(by=['reuse ratio'], inplace=True)

ax = sns.barplot(figure1_data, x='author', y='reuse ratio')
ax.set_ylabel("Reuse ratio")
ax.set_xlabel("Author")
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

save_image_in_all_formats(plt, 'figure1')


# Figure 2.

figure2_data = pd.read_csv('plots/final/figure2_data.csv')

f, axes = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(8,8))
sns.barplot(figure2_data[figure2_data['author'] == 'Hume'], x='volume', y='reuse ratio', ax=axes[0,0]).set(title='Hume', xlabel=None, ylim=(0, 0.6), ylabel='Reuse ratio')
sns.barplot(figure2_data[figure2_data['author'] == 'Rapin'], x='volume', y='reuse ratio', ax=axes[0,1]).set(title='Rapin', xlabel=None, ylim=(0, 0.6), ylabel='Reuse ratio')
sns.barplot(figure2_data[figure2_data['author'] == 'Guthrie'], x='volume', y='reuse ratio', ax=axes[1,0]).set(title='Guthrie', ylim=(0, 0.6), ylabel='Reuse ratio', xlabel='Volume')
sns.barplot(figure2_data[figure2_data['author'] == 'Carte'], x='volume', y='reuse ratio', ax=axes[1,1]).set(title='Carte', ylim=(0, 0.6), ylabel='Reuse ratio', xlabel='Volume')

for ax in axes.flat:
    # ax.set(xlabel='Volume', ylabel='Reuse ratio', ylim=(0, 0.6))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axes.flat:
#     ax.label_outer()

plt.tight_layout()
save_image_in_all_formats(plt, 'figure2')


# Figure 3.

figure3_data = pd.read_csv('plots/final/figure3_data.csv')
figure3_data['faction'] = figure3_data['faction'].str.capitalize()

ax = sns.histplot(
    data=figure3_data,
    x="volume", hue="faction",
    multiple="fill", stat="proportion", weights=figure3_data.proportion_factions,
    discrete=True, shrink=.8)

ax.set_ylabel("Proportion")
ax.set_xlabel("Volume")
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.get_legend().set_title("Faction")

save_image_in_all_formats(plt, 'figure3')


# Figure 4.

figure4_data = pd.read_csv('plots/final/figure4_data.csv')
figure4_data['header_text_short'] = figure4_data['header_text'].apply(get_shortened_header_text, max_length=50)

f, ax = plt.subplots(figsize=(5, 14))
sns.barplot(figure4_data, x='total_coverage_proportional',
            y='header_text_short', orient='h', hue='volume',
            palette=['grey', 'lightgrey'])

ax.set_xlabel("Proportion")
ax.set_ylabel("Chapter")
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.get_legend().set_title("Volume")

save_image_in_all_formats(plt, 'figure4', tight=True)


# Figure 5.

figure5_data = pd.read_csv('plots/final/figure5_data.csv')
figure5_data['faction'] = figure5_data['faction'].str.capitalize()
figure5_data['header_text_short'] = figure5_data['header_text'].apply(get_shortened_header_text, max_length=50)

f, ax = plt.subplots(figsize=(5, 14))
sns.barplot(figure5_data, x='percentage_of_part_length',
            y='header_text_short', orient='h', hue='faction',
            palette=['grey', 'lightgrey'])

ax.set_xlabel("Proportion")
ax.set_ylabel("Chapter")
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.get_legend().set_title("Faction")

save_image_in_all_formats(plt, 'figure5', tight=True)


# Figure 6.

figure6_data = pd.read_csv('plots/final/figure6_data.csv')
figure6_data['faction'] = figure6_data['faction'].str.capitalize()
figure6_data['header_text_short'] = figure6_data['header_text'].apply(get_shortened_header_text, max_length=50)

f, ax = plt.subplots(figsize=(5, 14))
sns.barplot(figure6_data, x='percentage_of_part_length',
            y='header_text_short', orient='h', hue='faction',
            palette=['grey', 'lightgrey'])

ax.set_xlabel("Proportion")
ax.set_ylabel("Chapter")
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.get_legend().set_title("Faction")

save_image_in_all_formats(plt, 'figure6', tight=True)


# Figure 7.

figure7_data = pd.read_csv('plots/final/figure7_data.csv')
figure7_data['faction'] = figure7_data['faction'].str.capitalize()
figure7_data['header_text_short'] = figure7_data['header_text'].apply(get_shortened_header_text, max_length=50)

f, ax = plt.subplots(figsize=(5, 4))
sns.barplot(figure7_data, x='percentage_of_part_length',
            y='header_text_short', orient='h', hue='faction')

ax.set_xlabel("Proportion")
ax.set_ylabel("Chapter")
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.get_legend().set_title("Faction")

save_image_in_all_formats(plt, 'figure7', tight=True)


# Figure 8.

figure8_data = pd.read_csv('plots/final/figure8_data.csv')
figure8_data['faction'] = figure8_data['faction'].str.capitalize()
figure8_data['header_text_short'] = figure8_data['header_text'].apply(get_shortened_header_text, max_length=50)

f, ax = plt.subplots(figsize=(5, 4))
sns.barplot(figure8_data, x='percentage_of_part_length',
            y='header_text_short', orient='h', hue='faction')

ax.set_xlabel("Proportion")
ax.set_ylabel("Chapter")
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.get_legend().set_title("Faction")

save_image_in_all_formats(plt, 'figure8', tight=True)


# Figure 9.

figure9_data = pd.read_csv('plots/final/figure9_data.csv')
figure9_data.sort_values('author_id', na_position='first', inplace=True)
figure9_data['header_text_short'] = figure9_data['header_text'].apply(get_shortened_header_text, max_length=50)

f, ax = plt.subplots(figsize=(10, 10))
sns.histplot(data=figure9_data,
             y='header_text_short', hue='author_name',
             multiple='stack', discrete=True, shrink=.8,
             weights=figure9_data.pages,
             )
ax.set_xlabel("Pages")
ax.set_ylabel("Chapter")

hatches = ['..', 'xx', '//', '\\\\', 'OO', '..', 'xx', '//', '\\\\', 'OO', None]*3
colors = ['white', 'grey', 'white', 'grey', 'white', 'grey', 'white', 'grey', 'white', 'grey', 'lightgrey']
hatch_colors = ['black', 'white', 'black', 'white', 'black', 'white', 'black', 'white', 'black', 'white', 'white']

for bars, hatch, color, hatch_color in zip(ax.containers, hatches, colors, hatch_colors):
    # Set a different hatch for each group of bars- and bar colour and hatch colour
    for bar in bars:
        bar.set_hatch(hatch)
        bar.set_facecolor(color)
        bar._hatch_color = mpl.colors.to_rgba(hatch_color)

hue_order = list(figure9_data['author_name'].drop_duplicates())
hue_order.reverse()
ax.legend(hue_order, handleheight=1.4, title='Author')

save_image_in_all_formats(plt, 'figure9', tight=True)
