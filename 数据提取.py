# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:04:12 2023

@author: hsliu
"""

import pandas as pd
import numpy as np

from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')
import datetime
from WindPy import w
import os
workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)
w.start()
#%%

'''
提取用于构建指数所需数据
1.数据提取
2.数据合并
'''

###1.数据提取
df = pd.read_excel("指标说明.xlsx",sheet_name='指标说明')
startdate='2010-01-01'
# startdate='2023-03-01'      #比最新一期时间早半年（宏观指标模板更新需要最近6月的宏观数据）
startdate_2 = '2011-02-01'  #计算同比要比startdate早一年（部分宏观数据要计算同比）
enddate='2023-11-10'        #最新一期时间
#%%提取数据
#提取月度预测数据
'''
M0061675	wind一致预测:工业增加值：当月同比
M0061679	wind一致预测:社会消费品零售总额：当月同比
M0061678	wind一致预测:固定资产投资完成额：累计同比
M0061681	wind一致预测:进口总额：当月同比
M0061680	wind一致预测:出口金额：当月同比
M0061683	wind一致预测：M2:同比
M0061684	wind一致预测：人民币贷款：同比
'''
edb_indicators1=df[(df['source']=='wind') & (df['freq']=='monthly')  & (df['备注1']=='wind一致预测')]['windcode']
data= w.edb(list(edb_indicators1), startdate, enddate)
Monthlydata1=pd.DataFrame(data.Data,index=data.Codes).T
Monthlydata1['date']=data.Times
Monthlydata1['date']=pd.to_datetime(Monthlydata1['date'])
Monthlydata1=Monthlydata1.sort_values(by=['date'])
Monthlydata1=Monthlydata1.fillna(method='pad')

#提取月度实际数据
'''
M0017126	PMI:同比
M6322780	产品销售率:累计同比
M5206730	社会融资规模：当月值
M5528822	抵押补充贷款（PSL)：期末余额
S2707411	70个大中城市新建商品住宅价格指数：当月同比
S2707396	30大中城市：商品房成交面积
S0029657	房地产开发投资完成额:累计同比
'''

edb_indicators2=df[(df['source']=='wind') & (df['freq']=='monthly')  & (df['备注1']!='wind一致预测')]['windcode']
data= w.edb(list(edb_indicators2), startdate_2, enddate)  #要计算同比，时间至少要1年
Monthlydata2=pd.DataFrame(data.Data,index=data.Codes).T
Monthlydata2['date']=data.Times
Monthlydata2['date']=pd.to_datetime(Monthlydata2['date'])
Monthlydata2=Monthlydata2.sort_values(by=['date'])

Monthlydata2['M5206730']=Monthlydata2['M5206730'].pct_change(periods=12)
Monthlydata2['S2707396']=Monthlydata2['S2707396'].pct_change(periods=12)
Monthlydata2['M5528822']=Monthlydata2['M5528822'].fillna(0)
Monthlydata2['M5528822']=Monthlydata2['M5528822']-Monthlydata2['M5528822'].shift(1)
Monthlydata2=Monthlydata2.fillna(method='pad')

#提取周度频率数据
'''
M0061614	公开市场逆回购：货币净投放
'''
edb_indicators2=df[(df['source']=='wind') & (df['freq']=='weekly') ]['windcode']
edb_indicators2=edb_indicators2.dropna()
data= w.edb(list(edb_indicators2), startdate, enddate)
Weeklydata=pd.DataFrame(data.Data,index=data.Codes).T
Weeklydata['date']=data.Times
Weeklydata['date']=pd.to_datetime(Weeklydata['date'])

Weeklydata=Weeklydata.sort_values(by=['date'])
Weeklydata=Weeklydata.fillna(method='pad')

#提取日度收益率数据（利率、汇率、股票收益率等数据）
'''
M0020194	上证综合指数:涨跌幅(不复权）
S0059744    1年中债国债到期收益率	
S0059749    10年中债国债到期收益率	
M0265873    中债企业债AAA净价指数	
M0265855    中债国开行债券总净价指数	
S0105896    南华综合指数	
M0149844	巨潮人民币实际汇率指数
M0096870	贷款市场报价利率（LPR）:1年
M0041371	逆回购利率：7天
'''
edb_indicators3=df[(df['source']=='wind') & (df['freq']=='daily') ]['windcode']
edb_indicators3=edb_indicators3.dropna()
data= w.edb(list(edb_indicators3), startdate, enddate)

Dailydata=pd.DataFrame(data.Data,index=data.Codes).T
Dailydata['date']=data.Times
Dailydata['date']=pd.to_datetime(Dailydata['date'])
Dailydata=Dailydata.sort_values(by=['date'])
Dailydata=Dailydata.fillna(method='pad')
Dailydata['year']=Dailydata['date'].dt.year
Dailydata['month']=Dailydata['date'].dt.month
Dailydata['week']=Dailydata['date'].dt.isocalendar().week
Dailydata['day']=Dailydata['date'].dt.day
Dailydata['term']=Dailydata['S0059749']-Dailydata['S0059744']    #计算期限利差
Dailydata['CS']=Dailydata['M0265873'].pct_change(periods=1)-Dailydata['M0265855'].pct_change(periods=1)   #计算信用利差
Dailydata['Commodity']=Dailydata['S0105896'].pct_change(periods=1)
Dailydata['M0149844']=Dailydata['M0149844']-Dailydata['M0149844'].shift(1)
Dailydata=Dailydata[Dailydata['date']>='2011-01-01']
#%%2.数据合并
#预测月度数据,不需要滞后
Monthlydata1['year']=Monthlydata1['date'].dt.year
Monthlydata1['month']=Monthlydata1['date'].dt.month
Monthlydata1=Monthlydata1.drop(['date'],axis=1)
total=pd.merge(Dailydata,Monthlydata1,on=['year','month'],how='inner')

#实际月度数据,滞后1个月
Monthlydata2['date']=Monthlydata2['date'].apply(lambda x: x-relativedelta(months=-1))
Monthlydata2['year']=Monthlydata2['date'].dt.year
Monthlydata2['month']=Monthlydata2['date'].dt.month
Monthlydata2=Monthlydata2.drop(['date'],axis=1)
total=pd.merge(total,Monthlydata2,on=['year','month'],how='inner')

#实际周度数据,滞后1周
Weeklydata=Weeklydata.sort_values(by=['date'])
Weeklydata['date']=Weeklydata['date'].apply(lambda x: x-relativedelta(weeks=-1))
Weeklydata['year']=Weeklydata['date'].dt.year
Weeklydata['week']=Weeklydata['date'].dt.isocalendar().week
Weeklydata=Weeklydata.drop(['date'],axis=1)

total=pd.merge(total,Weeklydata,on=['year','week'],how='inner')


total=total.sort_values(by=['date'])
total=total.fillna(method='pad')
total=total[total['date']>'2011-01-01']   #筛选2011年以来的数据
total=total.sort_values(by=['date'])
total=total.reset_index(drop=True)

total.to_excel('index_data_0821.xlsx')#导出数据备用
#%%TED利差

'''
提取用于ted预测数据,此为初次提取数据使用

'''

data=w.edb("G1147433", startdate, enddate)
Ted=pd.DataFrame(data.Data,columns=data.Times,index=data.Codes).T
Ted=Ted.reset_index()
Ted=Ted.rename(columns={'index':'date','G1147433':'ted'})
Ted=Ted.dropna()
Ted['date']=pd.to_datetime(Ted['date'])
Ted['year']=Ted['date'].dt.year
Ted['month']=Ted['date'].dt.month
Ted['day']=Ted['date'].dt.day
Ted=Ted[Ted['date']>'2011-01-01'] 

Ted.to_excel('Ted_data_0821.xlsx')#导出数据备用
#%%万得全A月度收益率序列
'''
提取股票指数月度收益率序列 ,此为初次提取数据使用
'''

data=w.wsd("881001.WI", "close", startdate, enddate, "")
stockindex=pd.DataFrame(data.Data,columns=data.Times).T
stockindex=stockindex.reset_index()
stockindex=stockindex.rename(columns={'index':'date',0:'close'})

stockindex.to_excel('stockindex_0821.xlsx')#导出数据备用
#%%更新ETF日度收益率
workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)

'''
提取ETF日度收益率序列 ,此为初次提取数据使用
'''
Assets= pd.read_excel("指标说明.xlsx",sheet_name='ETF')
Assets_indicators1=Assets[(Assets['source']=='wind')]['windcode']
Assets_indicators1=Assets_indicators1.tolist()
data=w.wsd(Assets_indicators1, "pct_chg", startdate, enddate, "")
Assets_dailyRET=pd.DataFrame(data.Data,index=data.Codes,columns=data.Times).T
Assets_dailyRET=Assets_dailyRET.reset_index()
Assets_dailyRET=Assets_dailyRET.rename(columns={'index':'date'})
Assets_dailyRET['date']=pd.to_datetime(Assets_dailyRET['date'])
Assets_dailyRET['year']=Assets_dailyRET['date'].dt.year
Assets_dailyRET['month']=Assets_dailyRET['date'].dt.month 
Assets_dailyRET['day']=Assets_dailyRET['date'].dt.day


Assets= pd.read_excel("指标说明.xlsx",sheet_name='ETF')
Assets_indicators1=Assets[(Assets['source']=='wind')]['windcode']
Assets_indicators1=Assets_indicators1.tolist()
data=w.wsd(Assets_indicators1, "close", startdate, enddate, "PriceAdj=B")   #PriceAdj=B表示后复权
Assets_dailyclose=pd.DataFrame(data.Data,index=data.Codes,columns=data.Times).T
Assets_dailyclose=Assets_dailyclose.reset_index()
Assets_dailyclose=Assets_dailyclose.rename(columns={'index':'date'})
Assets_dailyclose['date']=pd.to_datetime(Assets_dailyclose['date'])
Assets_dailyclose=pd.melt(Assets_dailyclose, id_vars='date', value_vars=Assets_dailyclose.iloc[:,1:])
Assets_dailyclose=Assets_dailyclose.rename(columns={'variable':'code','value':'close'})
Assets_dailyclose['date']=Assets_dailyclose['date'].apply(lambda dates: str(dates)[0:11].strip())

workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)
Assets_dailyRET.to_excel('dailyRET_0821.xlsx')#导出数据备用
Assets_dailyclose.to_excel('dailyclose_0821.xlsx')#导出数据备用

#%%将宏观数据导出到模板

'''
月频
供给侧
M0017126	PMI:同比
M6322780	产品销售率:累计同比
M0061675	wind一致预测:工业增加值：当月同比

需求侧
消费
M0061679	wind一致预测:社会消费品零售总额：当月同比
投资
M0061678	wind一致预测:固定资产投资完成额：累计同比
S2707411	70个大中城市新建商品住宅价格指数：当月同比
S2707396	30大中城市：商品房成交面积
S0029657	房地产开发投资完成额:累计同比
外贸
M0061681	wind一致预测:进口总额：当月同比
M0061680	wind一致预测:出口金额：当月同比

流动性
量价
M0061683	wind一致预测：M2:同比
M5206730	社会融资规模：当月值
M0061684	wind一致预测：人民币贷款：同比
货币政策
M5528822	抵押补充贷款（PSL)：期末余额
'''
workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)
factor_type = pd.read_excel("指标说明2.xlsx")
factor_type = factor_type.drop(['Unnamed: 0'], axis = 1)
#%%模板数据结果：月度数据
Monthlydata1['date'] = Monthlydata1.apply(lambda row : f"{int(row['year'])}年{int(row['month'])}月", axis = 1)
Monthlydata1 = Monthlydata1.T
Monthlydata1.columns = list(Monthlydata1.iloc[-1,:])
#Monthlydata1 = Monthlydata1.drop(['date'], axis = 0)
Monthlydata1.insert(0, 'code', Monthlydata1.index.values.tolist())
Monthlydata1.reset_index(drop = True, inplace = True)
Monthlydata1 = pd.merge(Monthlydata1, factor_type, left_on = 'code', right_on = 'indicators', how = 'left')
Monthlydata1 = Monthlydata1.dropna()
Monthlydata1 = Monthlydata1.drop(['indicators'], axis = 1)

Monthlydata2['date'] = Monthlydata2.apply(lambda row : f"{int(row['year'])}年{int(row['month'])}月", axis = 1)
Monthlydata2 = Monthlydata2.T
Monthlydata2.columns = list(Monthlydata2.iloc[-1,:])
#Monthlydata2 = Monthlydata2.drop(['date'], axis = 0)
Monthlydata2.insert(0, 'code', Monthlydata2.index.values.tolist())
Monthlydata2.reset_index(drop = True, inplace = True)
Monthlydata2 = pd.merge(Monthlydata2, factor_type, left_on = 'code', right_on = 'indicators', how = 'left')
Monthlydata2 = Monthlydata2.drop(['indicators'], axis = 1)
Monthlydata2_new = Monthlydata2.iloc[:-3,:]
Monthlydata2_new = Monthlydata2_new.dropna(axis = 1)

enddate_2 = datetime.datetime.strftime(datetime.datetime.strptime(enddate, "%Y-%m-%d") + relativedelta(months=1),
                                       "%Y-%m-%d")
month_list = month_list = list(pd.date_range(start=startdate,end=enddate_2,freq='M'))
month_list = month_list[-6:]
month_list = [str(d.year) + '年' + str(d.month) + '月' for d in month_list]
col_list = ['code'] + month_list + ['def', 'freq', 'first', 'second', 'third']

Monthlydata1_new = Monthlydata1[col_list].copy()
Monthlydata2_new = Monthlydata2_new[col_list].copy()
Monthlydata = pd.concat([Monthlydata1_new, Monthlydata2_new], axis = 0)
Monthlydata = Monthlydata.sort_values(by = ['first','second','third'])

workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)
Monthlydata.to_excel("宏观原始数据_月度数据_0822.xlsx", index = True)
#%%周度数据
edb_indicators2 = df[(df['source']=='wind') & (df['freq']=='weekly') ]['windcode']
edb_indicators2 = edb_indicators2.dropna()
data = w.edb(list(edb_indicators2), startdate, enddate)
Weeklydata = pd.DataFrame(data.Data,index=data.Codes).T
Weeklydata['date'] = data.Times
Weeklydata['date'] = pd.to_datetime(Weeklydata['date'])

Weeklydata = Weeklydata.sort_values(by=['date'])
Weeklydata = Weeklydata.fillna(method='pad')
Weeklydata = Weeklydata.iloc[-8:,:]
Weeklydata = Weeklydata.T
Weeklydata.columns = [datetime.datetime.strftime(x,"%Y-%m-%d") for x in list(Weeklydata.iloc[1,:])]
Weeklydata = Weeklydata.drop(['date'], axis = 0)
workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)
Weeklydata.to_excel("宏观原始数据_周度数据_0822.xlsx", index = True)
#%%日度数据
Dailydata1 = Dailydata.iloc[-21:,:]
Dailydata1 = Dailydata1.drop(['year', 'month', 'week','day'], axis = 1)
Dailydata1['date'] = Dailydata1['date'].apply(lambda x : datetime.datetime.strftime(x,"%Y-%m-%d"))
Dailydata1 = Dailydata1.T
Dailydata1.columns = list(Dailydata1.loc['date',:])
#Monthlydata1 = Monthlydata1.drop(['date'], axis = 0)
Dailydata1.insert(0, 'code', Dailydata1.index.values.tolist())
Dailydata1.reset_index(drop = True, inplace = True)
Dailydata1 = pd.merge(Dailydata1, factor_type, left_on = 'code', right_on = 'indicators', how = 'left')
Dailydata1 = Dailydata1.dropna()
Dailydata1 = Dailydata1.drop(['indicators'], axis = 1)

Dailydata1 = Dailydata1.sort_values(by = ['first','second','third'])
workspace=r'D:\YinheSecurity Intern Program\MacroTiming\代码'
os.chdir(workspace)
Dailydata1.to_excel("宏观原始数据_日度数据_0822.xlsx", index = True)
