import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = np.where((df['weight'] / (df['height'])**2) > 25, 1, 0)

# 3
df['cholesterol'] = np.where((df['cholesterol'])>1, 1, 0)
df['gluc'] = np.where((df['gluc'])>1, 1, 0)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    f_cat_counts = df_cat.groupby(['variable', 'value']).size().reset_index(name='total')
    
    # 7
    df_long = df.melt(id_vars='cardio', var_name='variable', value_name='value')

    # 8
    fig = sns.catplot(x='variable', y='total', hue='value', col='variable', data=f_cat_counts, kind='bar', height=4, aspect=1.5)


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(11, 9))

    # 15
    sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5, annot=True, fmt=".1f", ax=ax)


    # 16
    fig.savefig('heatmap.png')
    return fig
