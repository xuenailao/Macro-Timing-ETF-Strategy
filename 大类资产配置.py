# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 11:12:12 2023

@author: hsliu
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import riskfolio as rp
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from pypfopt import black_litterman
import warnings
warnings.filterwarnings('ignore')
import os

workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)
import gnBacktesting as bt
#%%函数1
def Standardization(data, poscols=None, negcols=None):   #poscols是正向指标name的list; negcols是负向指标name的list
    if poscols is None and negcols is None:
        return data
    elif poscols is not None and negcols is None:
        #如果只有正向指标，标准化结果z = (x - min) / (max - min)
        return (data[poscols] - data[poscols].min()) / (data[poscols].max() - data[poscols].min())
    elif poscols is None and negcols is not None:
        #如果只有负向指标，标准化指标x = (max - x) / (max - min)
        return (data[negcols].max - data[negcols]) / (data[negcols].max() - data[negcols].min())
    else:
        #如果既有正向指标又有负向指标，二者分别标准化，最后合并在一起
        #这样标准化后的结果都是数值越大越好
        a = (data[poscols] - data[poscols].min()) / (data[poscols].max() - data[poscols].min())
        b = (data[negcols].max() - data[negcols]) / (data[negcols].max() - data[negcols].min())
        return pd.concat([a, b], axis=1)


def WeightFun(data):    #熵值法计算权重
    K = 1 / np.log(len(data)) #log是以e为底数
    e = -K * np.sum(data * np.log(data))   #默认按列求和
    d = 1 - e
    weight = d / d.sum()
    return weight

def EntropyMethod(data, poscols=None, negcols=None):    #熵值法对指标加权获得经济指数/流动性指数
    """
        :param data:目标数据
        :param poscols: 正向指标列表，类型为列表或None
        :param negcols: 负向指标列表，类型为列表或None
        :return: index列即为熵值法所得综合指标
    """
    df = Standardization(data, poscols, negcols)
    df = df + 0.0001   #对于数值为0的数据进行非负平移 
    weight = WeightFun(df)
    df = df * weight
    df['index'] = df.sum(axis=1)
    return df['index']


def HPfilter(data, col='index', lamb=129600):  #Hodrick-Prescott (HP) 过滤器
    data['hp_cycle_' + col], data['hp_trend_' + col] = sm.tsa.filters.hpfilter(data[col], lamb=lamb)
    #根据Ravn and Uhlig(2002)的建议，对于年度数据lambda参数取值6.25(1600/4^4)，对于季度数据取值1600，对于月度数据取值129600(1600*3^4)
    #HP滤波适合提取经济周期（外生冲击）
    return data['hp_cycle_' + col]

#%%函数2
def markov_probability(df,proxy,if_output):
    '''
    Markov区制转换模型：
    时间序列存在于两个或多个状态，每个状态都有自己的概率分布
    马尔科夫转换自回归模型（MSAM）假定状态为“隐藏状态”，并假定潜在状态的的转换遵循同质一阶马尔可夫链，而下一个状态的概率仅取决于当前状态。
    可以通过最大似然法来估计从一个状态到下一个状态的转移概率，通过使似然函数最大化来估计参数值。
    -----------------------------------------------------------------
    参数解释
    k_regimes=2表示假设指数（经济指数、流动性指数、TED）有两种状态：高波动和低波动
    
    switching_variance默认是False，即时间序列的波动性（标准差）保持不变，这里考虑存在2种状态，因此设置为True。
    
    trend: Whether or not to include a trend. To include an intercept, time trend, or both, set trend=’c’,
           trend=’t’, or trend=’ct’. For no trend, set trend=’n’. Default is an intercept.
          
    order: The order of the model describes the dependence of the likelihood on previous regimes.
    '''
    if proxy == "econ":
        #econ分区
        #mod = sm.tsa.MarkovRegression(df['hp_cycle_econ'][1:], k_regimes=2, order=1, #trend='n', 
        #                              exog = df['hp_cycle_econ'][:-1],
        #                              switching_variance=True)
        mod = sm.tsa.MarkovRegression(df['hp_cycle_econ'], k_regimes=2, order=1, #trend='n', 
                                      switching_variance=True)
        #mod = sm.tsa.MarkovRegression(df['hp_cycle_econ'].diff(1), k_regimes=2, order=1, #trend='n', 
        #                              switching_variance=True)
        res = mod.fit()
        #print(len(df))
        #print(len(res.smoothed_marginal_probabilities[1]))
        df['high'] = res.smoothed_marginal_probabilities[1]   #高波动regime概率
        df['low'] = res.smoothed_marginal_probabilities[0]    #低波动regime概率
        '''
        if if_output:   #如果if_output为True，输出结果
            print(res.summary())
            print('经济周期duration：', res.expected_durations)
            '''
        
    elif proxy == "Finance":
         # cpi分区
         #mod = sm.tsa.MarkovRegression(df['hp_cycle_Finance'][1:], k_regimes=2, order=1, #trend='n',   #trend = 'n'
         #                              exog = df['hp_cycle_Finance'][:-1],
         #                              switching_variance=True)
         mod = sm.tsa.MarkovRegression(df['hp_cycle_Finance'], k_regimes=2, order=1, #trend='n',   #trend = 'n'
                                       switching_variance=True)
         #mod = sm.tsa.MarkovRegression(df['hp_cycle_Finance'].diff(1), k_regimes=2, #order=1, #trend='n',   #trend = 'n'
         #                              switching_variance=True)
         res = mod.fit()
         #print(len(df))
         #print(res.smoothed_marginal_probabilities[1])
         df['high'] = res.smoothed_marginal_probabilities[1]
         df['low'] = res.smoothed_marginal_probabilities[0]
         '''
         if if_output:   #如果if_output为True，输出结果
             print(res.summary())
             print('流动性周期duration：', res.expected_durations)'''
         
    elif proxy == "ted":
         # ted分区
         #mod = sm.tsa.MarkovRegression(df['hp_cycle_ted'][1:], k_regimes=2, order=1, #trend='n',   #trend = 'n'
         #                              exog = df['hp_cycle_ted'][:-1],
         #                              switching_variance=True)
         mod = sm.tsa.MarkovRegression(df['hp_cycle_ted'], k_regimes=2, order=1, #trend='n',   #trend = 'n'
                                       switching_variance=True)
         #mod = sm.tsa.MarkovRegression(df['hp_cycle_ted'].diff(1), k_regimes=2, #order=1, #trend='n',   #trend = 'n'
         #                              switching_variance=True)
         res = mod.fit()
         df['high'] = res.smoothed_marginal_probabilities[1]
         df['low'] = res.smoothed_marginal_probabilities[0]
         '''
         if if_output:   #如果if_output为True，输出结果
             print(res.summary())
             print('TED周期duration：', res.expected_durations)'''
         
    return df


#%%函数3
def judge_state_Daily(df,proxy,if_output):
    df['trend'] = HPfilter(df,proxy)
    df = markov_probability(df,proxy,if_output)    #用Morkov Regime Switching Dynamic Model获得每天处于两种regime的概率
    df['up']=df.apply(lambda x: 1 if x['high']>x['low'] else 0, axis=1)     #如果高波动概率大于低波动概率，说明处于上行周期
    df['down']=df.apply(lambda x: 1 if x['high']<x['low'] else 0, axis=1)   #反之处于下行周期
    return df

def judge_state_Monthly(df,proxy):
   
    if proxy == "econ":
        stat1 = df.groupby(['year','month'])['up'].sum()    #当月处于上行周期的天数
        stat1 = stat1.reset_index()
        stat2 = df.groupby(['year','month'])['down'].sum()  #当月处于下行周期的天数
        stat2 = stat2.reset_index()
        state = pd.merge(stat1,stat2,on=['year','month'],how='inner')
        state['econ_up'] = state.apply(lambda x: 1 if x['up']>x['down'] else 0, axis=1)     #如果上行天数大于下行，说明处于上行周期
        state['econ_down'] = state.apply(lambda x: 1 if x['up']<=x['down'] else 0, axis=1)  #反之处于下行周期
        df = pd.merge(df,state[['year','month','econ_up','econ_down']],on=['year','month'],how='inner')  #输出周期判断结果
    
    elif proxy == "Finance":
        stat1=df.groupby(['year','month'])['up'].sum()
        stat1=stat1.reset_index()
        stat2=df.groupby(['year','month'])['down'].sum()
        stat2=stat2.reset_index()
        state=pd.merge(stat1,stat2,on=['year','month'],how='inner')
        state['inf_up']=state.apply(lambda x: 1 if x['up']>x['down'] else 0, axis=1)
        state['inf_down']=state.apply(lambda x: 1 if x['up']<=x['down'] else 0, axis=1)
       
        df=pd.merge(df,state[['year','month','inf_up','inf_down']],on=['year','month'],how='inner')
    
       
    elif proxy == "ted":
        stat1=df.groupby(['year','month'])['up'].sum()
        stat1=stat1.reset_index()
        stat2=df.groupby(['year','month'])['down'].sum()
        stat2=stat2.reset_index()
        state=pd.merge(stat1,stat2,on=['year','month'],how='inner')
        state['sp_up']=state.apply(lambda x: 1 if x['up']>x['down'] else 0, axis=1)
        state['sp_down']=state.apply(lambda x: 1 if x['up']<=x['down'] else 0, axis=1)
        
        df=pd.merge(df,state[['year','month','sp_up','sp_down']],on=['year','month'],how='inner')
    
    return df



''' 
def HPfilter(data, col='index', lamb=129600):
    data['hp_cycle_' + col], data['hp_trend_' + col] = sm.tsa.filters.hpfilter(data[col], lamb=lamb)
    return data['hp_cycle_' + col]


def assetweights(df,proxy):
    
    if proxy == "econ":
        #econ分区
        mod = sm.tsa.MarkovRegression(df['hp_cycle_econ'], k_regimes=2, trend='c', order=1, switching_variance=True)
        res = mod.fit()#WTo include an intercept, time trend, or both, set trend=’c’, trend=’t’, or trend=’ct’. For no trend, set trend=’n’. Default is an intercept.
        df['high'] = res.smoothed_marginal_probabilities[1]
        df['low'] = res.smoothed_marginal_probabilities[0]
        
    elif proxy == "Finance":
         # cpi分区
         mod = sm.tsa.MarkovRegression(df['hp_cycle_Finance'], k_regimes=2, trend='n', order=1, switching_variance=True)
         res = mod.fit()
         df['high'] = res.smoothed_marginal_probabilities[1]
         df['low'] = res.smoothed_marginal_probabilities[0]
         
    elif proxy == "ted":
         # ted分区
         mod = sm.tsa.MarkovRegression(df['hp_cycle_ted'], k_regimes=2, trend='n', order=1, switching_variance=True)
         res = mod.fit()
         df['high'] = res.smoothed_marginal_probabilities[1]
         df['low'] = res.smoothed_marginal_probabilities[0]
         
    return df


def judge_state(df,proxy,datestart):
  
    df['trend'] = HPfilter(df,proxy)
    assetweights(df,proxy)
    df['up']=df.apply(lambda x: 1 if x['high']>x['low'] else 0, axis=1)
    df['down']=df.apply(lambda x: 1 if x['high']<x['low'] else 0, axis=1)
    df=df[df['date']>datestart]
    if df['up'].sum()>df['down'].sum():
        state=1
    else:
        state=0
    return state

'''


def judge_econ_state(df):
    """
    复苏：经济上行，流动性宽松
    过热：经济上行，流动性紧缩
    滞胀：经济下行，流动性紧缩
    衰退：经济下行，流动性宽松
    """
    df['state'] = np.where(np.logical_and(df['econ_up'] == 1, df['inf_up'] == 1), 1, np.nan) # 复苏
    df['state'] = np.where(np.logical_and(df['econ_up'] == 1, df['inf_up'] == 0), 2, df['state']) # 过热
    df['state'] = np.where(np.logical_and(df['econ_up'] == 0, df['inf_up'] == 0), 3, df['state']) # 滞胀
    df['state'] = np.where(np.logical_and(df['econ_up'] == 0, df['inf_up'] == 1), 4, df['state']) # 衰退
    return df

def bl(Assets_RET,state,delta):   #Black Litterman模型计算大类资产权重
    df=pd.merge(Assets_RET,state,on=['year','month','day'],how='left')
    df=df.dropna()
    df=df.reset_index(drop=True)
    
    constraints1 = pd.read_excel('con1.xlsx') # 复苏约束表
    constraints2 = pd.read_excel('con2.xlsx') # 过热约束表
    constraints3 = pd.read_excel('con3.xlsx') # 滞胀约束表
    constraints4 = pd.read_excel('con4.xlsx')  # 衰退约束表
    if df.iloc[-1]['sp_up'] == 0:#不配置美股情况
        asset_classes = pd.read_excel('a1.xlsx')
        assets = asset_classes['Assets']
        mean = df.groupby(by=['state']).mean()[assets].T    #如果历史不包括状态，则无法估计
        port = rp.Portfolio(returns = df[assets])      # 实例化一个投资组合，输入为收益率数据df[assets]
        P = np.eye(len(assets))
        Q = np.array(mean[df.iloc[-1]['state']]).reshape(-1, 1)    #转化为1列
        port.blacklitterman_stats(P=P, Q=Q, delta=delta, eq=True, method_mu='hist', method_cov='hist')
        if df.iloc[-1]['state'] == 1:   #根据经济周期选择constrains
            constraints = constraints1
        elif df.iloc[-1]['state']== 2:
            constraints = constraints2
        elif df.iloc[-1]['state'] == 3:
            constraints = constraints3
        else:
            constraints = constraints4
        C, D = rp.assets_constraints(constraints, asset_classes)
        port.ainequality = C
        port.binequality = D
        weight = port.optimization(model='BL', rm='MV', obj='Sharpe', rf=0, l=delta, hist=False)  # 传入参数，求解权重
     
    else: #配置美股情况
        asset_classes = pd.read_excel('a2.xlsx')
        assets = asset_classes['Assets'] 
        mean = df.groupby(by=['state']).mean()[assets].T###如果历史不包括状态，则无法估计
        port = rp.Portfolio(returns=df[assets])
        P = np.eye(len(assets))
        Q = np.array(mean[df.iloc[-1]['state']]).reshape(-1, 1)#转化为1列
        port.blacklitterman_stats(P=P, Q=Q, delta=delta, eq=True, method_mu='hist', method_cov='hist')
        if df.iloc[-1]['state'] == 1:   #根据经济周期选择constrains
            constraints = constraints1
        elif df.iloc[-1]['state']== 2:
            constraints = constraints2
        elif df.iloc[-1]['state'] == 3:
            constraints = constraints3
        else:
            constraints = constraints4
        C, D = rp.assets_constraints(constraints, asset_classes)
        port.ainequality = C
        port.binequality = D
        weight = port.optimization(model='BL', rm='MV', obj='Sharpe', rf=0, l=delta, hist=False)
    return  weight


###分区：每日分区

def asset_stateweight(dailyRET,index_data1,index_data2,Ted_data,stockindex,output_filename):
    finance_dailydata = pd.DataFrame() #储存流动性指数日度数据
    econ_dailydata = pd.DataFrame() #储存经济指数日度数据
    ted_dailydata = pd.DataFrame()#储存ted日度数据
    #state_dailydata= pd.DataFrame() #储存日状态数据
    
    state_data = pd.DataFrame() #储存月度经济状态
    weight_data = pd.DataFrame() #储存权重

    datelist=dailyRET[(dailyRET['date']>'2014-01-01')]  #原来是2020-01-01
    datelist=datelist.sort_values(by=['date'])
    datelist['rank']=datelist.groupby(['year','month'])['day'].rank(method='min',ascending=False)
    datelist=datelist[datelist['rank']==1]['date'].tolist()
   
    for i in datelist:
        
        #判断是否为最后一日并决定是否输出结果
        if i == datelist[-1]:
            if_output = True
        else:
            if_output = False
        
        #货币信贷指数构建及分区，指数为日度数据
        temp1 = index_data1[index_data1['date']<=i]    #选取i日之前的数据
        temp1['Finance'] = EntropyMethod(temp1, poscols=poscols1,negcols=negcols1)   #计算得到流动性指数
        temp1 = judge_state_Daily(temp1,'Finance',if_output)  #判断日状态
        temp1 = judge_state_Monthly(temp1,'Finance')  #判断月状态
       
        finance_dailydata=pd.concat([finance_dailydata,temp1[(temp1['year']==int(i.strftime("%Y")))  & (temp1['month']==int(i.strftime("%m")))]])#只保留当月
        
       
        #经济指数构建及分区，指数为日度数据
        temp2 = index_data2[index_data2['date']<=i] 
        temp2['econ'] = EntropyMethod(temp2, poscols=poscols2,negcols=negcols2)
        temp2 = judge_state_Daily(temp2,'econ',if_output)
        temp2 = judge_state_Monthly(temp2,'econ')
       
        econ_dailydata=pd.concat([econ_dailydata, temp2[(temp2['year']==int(i.strftime("%Y")))  & (temp2['month']==int(i.strftime("%m")))]])
        
        
        #ted为日度数据
        temp3=Ted_data[Ted_data['date']<=i]
        temp3=judge_state_Daily(temp3,'ted',if_output)
        temp3=judge_state_Monthly(temp3,'ted')
      
        ted_dailydata=pd.concat([ted_dailydata,temp3[(temp3['year']==int(i.strftime("%Y")))  & (temp3['month']==int(i.strftime("%m")))]])
        
        
        #经济状态
        state=pd.merge(temp1[['date','year','month','day','inf_up','inf_down']],
                       temp2[['year','month','day','econ_up','econ_down']],on=['year','month','day'],how='inner')
        state=judge_econ_state(state)
        state=pd.merge(state,temp3[['year','month','day','sp_up','sp_down']],on=['year','month','day'],how='inner')
   
        state['maxday']=state['date'].agg('max')
        state_data=pd.concat([state_data,state[state['date']==state['maxday']]])
        
        # #求大类资产权重
        # temp4 = stockindex[stockindex['date']<=i]
        # delta = black_litterman.market_implied_risk_aversion(temp4['close'], frequency=252,
        #                                                     risk_free_rate=0)    #使用Wind全A指数收盘价求风险厌恶系数delta
        # temp5 = dailyRET[dailyRET['date'] <= i].iloc[:,1:]
        # weight = bl(temp5,state,delta)    #用Black Litterman模型计算权重
        # weight = weight.T
        # weight['date'] = i
        # weight_data = pd.concat([weight_data,weight])   #将新的weight结果添加到weight_data中
        
        
    finance_dailydata.to_excel(output_filename[0] + '.xlsx') 
    econ_dailydata.to_excel(output_filename[1] + '.xlsx')
    ted_dailydata.to_excel(output_filename[2] + '.xlsx')    
    
    
    state_data.to_excel(output_filename[3] + '.xlsx')
    # weight_data.to_excel(output_filename[4] + '.xlsx')    
    return  state_data


#%%宏观指标区分货币信贷指标、经济指标，获得正向指标和负向指标的list

workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)
   
'''
宏观指标区分货币信贷指标、经济指标

'''

df = pd.read_excel("指标说明.xlsx",sheet_name='指标说明')
Allindicators=df['indicators'].dropna().tolist()#所有宏观指标
dateindicator=['date','year','month','week','day']

indicator1=df[(df['type']=='money') | (df['type']=='Finance') | (df['type']=='policy')]['windcode']
indicator1=indicator1.tolist()#货币信贷指标

indicator2=set(Allindicators).difference(set(indicator1))
indicator2=list(indicator2)#经济指标

indicator1=indicator1+dateindicator
indicator2=indicator2+dateindicator

'''
区分指标方向：正向指标、负向指标
'''

negcols=df[(df['符号']=='负向')]['indicators'].tolist()#所有负向宏观指标

negcols1 = list(set(negcols).intersection(set(indicator1)))#货币信贷负向指标
poscols1 = list(set(indicator1).difference(set(negcols1),set(dateindicator))) #货币信贷正向指标

negcols2 = list(set(negcols).intersection(set(indicator2))) #经济负向指标
poscols2 = list(set(indicator2).difference(set(negcols2),set(dateindicator)))  #经济正向指标

#%%读取指数数据
# 读取指数数据
index_data=pd.read_excel('index_data.xlsx')
index_data=index_data.iloc[:,1:]
index_data=index_data.dropna()
index_data_new = pd.read_excel('index_data_0821.xlsx')    #把两个数据合并
index_data_new = index_data_new.iloc[:,1:]
index_data_new = index_data_new.dropna()
index_data_new = index_data_new[index_data_new['date'] > '2023-07-20']
index_data = pd.concat([index_data, index_data_new], axis = 0)
index_data.reset_index(drop = True, inplace = True)

index_data1=index_data[indicator1]
index_data2=index_data[indicator2]

Ted_data=pd.read_excel('Ted_data.xlsx')
Ted_data=Ted_data.iloc[:,1:]
Ted_data_new = pd.read_excel('Ted_data_0821.xlsx')   #把两个数据合并
Ted_data_new = Ted_data_new.iloc[:,1:]
Ted_data_new = Ted_data_new[Ted_data_new['date'] > '2023-07-20']
Ted_data = pd.concat([Ted_data, Ted_data_new], axis = 0)
Ted_data.reset_index(drop = True, inplace = True)

stockindex = pd.read_excel('stockindex.xlsx')
stockindex = stockindex.iloc[:,1:]
stockindex_new = pd.read_excel('stockindex_0821.xlsx')   #把两个数据合并
stockindex_new = stockindex_new.iloc[:,1:]
stockindex_new = stockindex_new[stockindex_new['date'] > '2023-07-20']
stockindex = pd.concat([stockindex, stockindex_new], axis = 0)
stockindex.reset_index(drop = True, inplace = True)
#%%计算大类资产权重：获取资产收益率（用这里的代码）
'''
大类资产权重
'''
workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)

dailyRET1 = pd.read_excel('dailyRET_0821.xlsx')     #把两个数据合并
dailyRET1 = dailyRET1.iloc[:,1:]
dailyRET1 = dailyRET1[dailyRET1['date'] > '2023-07-20']
dailyRET2 = pd.read_excel('dailyRET.xlsx')#可以输入不同的资产收益率
dailyRET2 = dailyRET2.iloc[:,1:]
dailyRET = pd.concat([dailyRET2,dailyRET1], axis = 0)
dailyRET.reset_index(drop = True, inplace = True)
dailyRET=dailyRET[dailyRET['date']>='2013-01-01']
#%%计算大类资产权重：计算权重
output_filename = ['finance_dailydata','econ_dailydata','ted_dailydata','state_data','weight_data']
output_filename = [x + '_0821' for x in output_filename]
#计算大类资产的权重
state_data = asset_stateweight(dailyRET,index_data1,index_data2,Ted_data,stockindex,output_filename)
#%%回测
workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)
dailyClose = pd.read_excel('dailyClose.xlsx')#可以输入不同的资产收盘价
dailyClose = dailyClose.iloc[:,1:]
dailyClose = dailyClose[dailyClose['date']>'2020-06-30']
dailyClose_new = pd.read_excel('dailyClose_0821.xlsx')      #把两个数据合并
dailyClose_new = dailyClose_new.iloc[:,1:]
dailyClose_new = dailyClose_new[dailyClose_new['date']>'2023-07-20']
dailyClose = pd.concat([dailyClose, dailyClose_new], axis = 0)
dailyClose.reset_index(drop = True, inplace = True)

#weight_data=pd.read_excel('weight_data.xlsx')#可以输入不同的资产收盘价
#weight_data=weight_data.iloc[:,1:]

weight_data=weight_data.reset_index(drop=True)    
weight_data=weight_data.fillna(0)
weight_data=weight_data[weight_data['date']>'2020-05-31']
weight_data=pd.melt(weight_data, id_vars='date', value_vars=list(set(dailyClose['code'])))
weight_data=weight_data.rename(columns={'variable':'code','value':'w'})
weight_data['date']=weight_data['date'].apply(lambda dates: str(dates)[0:11].strip())
#%%
###BL回测
oi=r"D:\YinheSecurity Intern Program\MacroTiming\代码"
es = bt.gnStatement(oi,dailyClose,"BL_0821")   #输出结果
es.backtest(weight_data, "ETF", False)