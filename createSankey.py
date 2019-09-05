"""
Code is adapted from https://gist.github.com/ken333135/09f8793fff5a6df28558b17e516f91ab
"""
import pandas as pd
import plotly
import chart_studio.plotly as py
import joblib


def genSankey(df, cat_cols=[], value_cols='', title='Sankey Diagram'):
    """ max number of cols -> 6 colours
    """
    colourPallet = ['#4B8BBE', '#306998',
                    '#FFE873', '#FFD43B', '#646464']  # change
    labelList = []
    colourNumList = []
    for catCol in cat_cols:
        labelListTemp = list(set(df[catCol].values))
        colourNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp

    # Remove dublicats from labelList
    labelList = list(dict.fromkeys(labelList))

    # define colors based on number of levels
    colourList = []
    for idx, colourNum in enumerate(colourNumList):
        colourList = colourList + [colourPallet[idx]] * colourNum

    # transform df into a source-target pair
    for i in range(len(cat_cols) - 1):
        if i == 0:
            sourceTargetDf = df[[cat_cols[i], cat_cols[i + 1], value_cols]]
            sourceTargetDf.columns = ['source', 'target', 'count']
        else:
            tempDF = df[[cat_cols[i], cat_cols[i + 1], value_cols]]
            tempDF.columns = ['source', 'target', 'count']
            sourceTargetDf = pd.concat([sourceTargetDf, tempDF])
        sourceTargetDf = sourceTargetDf.groupby(
            ['source', 'target']).agg({'count': 'sum'}).reset_index()
    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(
        lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(
        lambda x: labelList.index(x))

    # Creating the sankey diagram
    data = dict(
        type='sankey',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(
                color='black',
                width=0.5
            ),
            label=labelList,
            color=colourList
        ),
        link=dict(
            source=sourceTargetDf['sourceID'],
            target=sourceTargetDf['targetID'],
            value=sourceTargetDf['count']
        )
    )
    layout = dict(
        title=title,
        font=dict(
            size=10
        )
    )
    fig = dict(data=[data], layout=layout)
    return fig
