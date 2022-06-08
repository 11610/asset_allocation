#!/usr/bin/env python
# coding: utf-8

# In[1]:


#导入相关库
from WindPy import w
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from scipy.optimize import curve_fit
from scipy.optimize import brent, fmin, minimize
from tqdm import trange
w.start()


# # 数据处理函数

# In[2]:


#处理累计数据（即差分）
def cumdata_clean(data):
    result,name,index,data = [],data.columns,data.index,data.iloc[:,0].tolist()
    for i in range(len(index)):
        if index[i].month == 2:
            result.append(data[i]/2)
            result.append(data[i]/2)
        elif index[i].month != 2:
            result.append(data[i] - data[i-1])
    return pd.DataFrame(result,index = pd.date_range("2008-01-31",periods = len(result), freq="M"),columns = name)


# In[3]:


#检验数据是否有季节性
def check(data):
    result = []
    name = data.columns
    data = data.reset_index()
    data = data.dropna()
    data['month'] = data['date'].apply(lambda x: x.month)
    anova = []
    for i in range(1,13):
        anova.append(data[data['month'] == i].iloc[:,1].tolist())
    f,p = scipy.stats.f_oneway(*anova)
    result.append([f,p])
    f,p = scipy.stats.kruskal(*anova)
    result.append([f,p])
    result = pd.DataFrame(result,columns = ['F-value','p-value'])
    return result


# In[4]:


#HP滤波
def HP(y, lamb):
    def D_matrix(N):
        D = np.zeros((N-1,N))
        D[:,1:] = np.eye(N-1)
        D[:,:-1] -= np.eye(N-1)
        return D
    N = len(y)
    D1 = D_matrix(N)
    D2 = D_matrix(N-1)
    D = D2 @ D1
    g = np.linalg.inv((np.eye(N)+lamb*D.T@D))@ y
    return g


# In[5]:


#去除季节性&HP滤波提取循环项
def data_clean(data,name,seasonal,lamb):  
    if seasonal == 1:
        res = sm.tsa.seasonal_decompose(data,period=12,model="multiplicative",extrapolate_trend = 'freq')
        data.columns = [name]
        data_compose = data.join([res.resid,res.seasonal])
        data_compose['T&C'] = data_compose[name]/(data_compose['resid']*data_compose['seasonal'])
        data_compose['trend_HP'] = pd.DataFrame(HP(data_compose['T&C'],lamb),index = data_compose.index,columns = ['trend_HP'])
        data_compose['Circle_HP'] = data_compose['T&C']/data_compose['trend_HP']
        data_compose['Circle_HP_pct'] = data_compose['Circle_HP'].pct_change(periods = 12)
    
    elif seasonal == 0:
        data.columns = [name]
        data_compose = data
        data_compose['trend_HP'] = pd.DataFrame(HP(data_compose[name],lamb),index = data_compose.index,columns = ['trend_HP'])
        data_compose['Circle_HP'] = data_compose[name]/data_compose['trend_HP']
        data_compose['Circle_HP_pct'] = data_compose['Circle_HP'].pct_change(periods = 12)
    
    return data_compose['Circle_HP_pct']


# # 合成增长因子

# In[6]:


ID_name = ['产量:发电量', '产量:铝材', '产量:硫酸', '产量:乙烯', '产量:空调', '产量:汽车', '销量:叉车:全行业', '货物周转量总计', '税收收入', '房屋新开工面积', '房地产开发投资完成额']
ID_code = ['S0027012', 'S0027571', 'S0027103', 'S0027159', 'S0028202', 'S0027907', 'S6001740', 'S0036018', 'M0024057', 'S0029669', 'S0029656']


# In[7]:


PMI = ((w.edb('M0017126',"20060131","","Fill=Previous",usedf = True)[1] - 50)/100 + 1).cumprod()
PMI = pd.DataFrame(data_clean(PMI,'PMI',0,1440))
PMI.columns = ['PMI']


# In[8]:


#经济增长相关指标处理
from tqdm import trange
for i in trange(len(ID_code)):
    if i < len(ID_code) - 2:
        data = w.edb(ID_code[i],"20060101","","Fill=Previous",usedf = True)[1]
        data['date'] = pd.to_datetime(data.index)
        data.set_index('date', inplace=True)
        data = data.resample('M').mean()
        data_index = data.index
        
    if i >= len(ID_code) - 2:
        data = w.edb(ID_code[i],"20060101","","Fill=Previous",usedf = True)[1]
        data = cumdata_clean(data)
        data['date'] = pd.to_datetime(data.index)
        data.set_index('date', inplace=True)
        data = data.resample('M').mean()
        data_index = data.index
        
    if check(data)['p-value'].mean() < 0.1:
        data = pd.DataFrame(data_clean(data.dropna(),ID_name[i],1,14400),index = data_index)
    else:
        data = pd.DataFrame(data_clean(data.dropna(),ID_name[i],0,14400),index = data_index)
    data.columns = [ID_name[i]]
    
    if i == 0:
        result = PMI.join(data,how = 'left')
    elif i != 0:
        result = result.join(data,how = 'left')


# In[9]:


result = result.iloc[12:,:]
result_ = pd.DataFrame(result['PMI'])
result_.columns = ['PMI']
for i in range(1,12):  
    result_ = result_.join(result.iloc[:,i].fillna(value = result['PMI'],axis = 0))


# In[10]:


#OCED法合成指标
SDj = abs(result_.iloc[:,1:] - result_.iloc[:,1:].mean()).sum()/result_.shape[0]
SCj = (result_.iloc[:,1:] - result_.iloc[:,1:].mean())/SDj
S = SCj.T.sum()
X = PMI.iloc[12:,]
k = abs(X - X.mean()).sum()[0]/abs(S - S.mean()).sum()
d = X.mean()[0] - S.mean()
CI_increase = k*S + d


# In[11]:


plt.style.use('ggplot')
plt.figure(figsize=(15,9),dpi = 750)
plt.plot(CI_increase,label = 'increase')


# # 合成通胀因子

# In[12]:


ID_code = ['S0066840', 'S0031507', 'S5711190', 'S0031525', 'S5705040']
ID_name = ['大宗价:猪肉', 'CRB 现货指数:油脂', '螺纹价格指数', '期货结算价:布伦特原油', 'MyIpic 矿价指数:综合']


# In[13]:


CPI = w.edb('M0000705',"20060131","","Fill=Previous",usedf = True)[1]
PPI = w.edb('M0049160',"20060131","","Fill=Previous",usedf = True)[1]
CPI = (CPI/100 + 1).cumprod()
PPI = (PPI/100 + 1).cumprod()
CPI_weight = 1/CPI.std() / (1/CPI.std() + 1/PPI.std())
PPI_weight = 1/PPI.std() / (1/CPI.std() + 1/PPI.std())
inflation = CPI*CPI_weight + PPI*PPI_weight
inflation = data_clean(inflation.dropna(),'inflation',0,14400)
inflation = pd.DataFrame(inflation)
inflation.columns = ['inflation']


# In[14]:


#通胀相关指标处理
from tqdm import trange
for i in trange(len(ID_code)):
    data = w.edb(ID_code[i],"20060101","","Fill=Previous",usedf = True)[1]
    data['date'] = pd.to_datetime(data.index)
    data.set_index('date', inplace=True)
    data = data.resample('M').mean()
    data_index = data.index
    if check(data)['p-value'].mean() < 0.1:
        data = pd.DataFrame(data_clean(data.dropna(),ID_name[i],1,14400),index = data_index)
    elif check(data)['p-value'].mean() >= 0.1:
        data = pd.DataFrame(data_clean(data.dropna(),ID_name[i],0,14400),index = data_index)
    data.columns = [ID_name[i]]
    
    if i == 0:
        result = inflation.join(data,how = 'left')
    elif i != 0:
        result = result.join(data,how = 'left')


# In[15]:


result = result.iloc[12:,:]
result_ = pd.DataFrame(result['inflation'])
result_.columns = ['inflation']
for i in range(1,6):  
    result_ = result_.join(result.iloc[:,i].fillna(value = result['inflation'],axis = 0))


# In[16]:


#OCED法合成指标
SDj = abs(result_.iloc[:,1:] - result_.iloc[:,1:].mean()).sum()/result_.shape[0]
SCj = (result_.iloc[:,1:] - result_.iloc[:,1:].mean())/SDj
S = SCj.T.sum()
X = inflation.iloc[12:,]
k = abs(X - X.mean()).sum()[0]/abs(S - S.mean()).sum()
d = X.mean()[0] - S.mean()
CI_inflation = k*S + d


# In[17]:


plt.figure(figsize=(15,9),dpi = 750)
plt.plot(CI_inflation,label = 'inflation')


# # 合成信用因子

# In[18]:


ID_name = ['M1', 'M2', '社会融资规模:当月值', '金融机构:各项贷款余额', '金融机构:企业存款余额']
ID_code = ['M0001382', 'M0001384', 'M5206730', 'M0009969', 'M0043410']


# In[19]:


#信用相关指标处理
from tqdm import trange
for i in trange(len(ID_code)):
    data = w.edb(ID_code[i],"20060131","","Fill=Previous",usedf = True)[1]
    data['date'] = pd.to_datetime(data.index)
    data.set_index('date', inplace=True)
    data = data.resample('M').mean()
    data_index = data.index
    if check(data)['p-value'].mean() < 0.1:
        data = pd.DataFrame(data_clean(data.dropna(),ID_name[i],1,14400),index = data_index)
    elif check(data)['p-value'].mean() >= 0.1:
        data = pd.DataFrame(data_clean(data.dropna(),ID_name[i],0,14400),index = data_index)
    data.columns = [ID_name[i]]
    
    if i == 0:
        result = inflation.join(data,how = 'left')
    elif i != 0:
        result = result.join(data,how = 'left')


# In[20]:


result_ = result.iloc[12:,:]


# In[21]:


#OCED法合成指标
SDj = abs(result_ - result_.mean()).sum()/result_.shape[0]
SCj = (result_ - result_.mean())/SDj
S = SCj.T.sum()
CI_credit = S


# In[22]:


plt.figure(figsize=(15,9),dpi = 750)
plt.plot(CI_credit)


# # 合成货币因子

# In[23]:


ID_code = ['S0059744','M1001795','M0017142']
ID_name = ['国债到期收益率:1年','R007','SHIBOR:3个月']


# In[24]:


#货币相关指标处理
for i in trange(len(ID_code)):
    data = w.edb(ID_code[i],"20060101","","Fill=Previous",usedf = True)[1]
    data['date'] = pd.to_datetime(data.index)
    data.set_index('date', inplace=True)
    data = data.resample('M').mean()
    data_index = data.index
    data = HP(data,1)
    data.index = data_index
    data.columns = [ID_name[i]]
    
    if i == 0:
        result = inflation.join(data,how = 'left')
    elif i != 0:
        result = result.join(data,how = 'left')


# In[25]:


result_ = result.iloc[12:,1:]


# In[26]:


#OCED法合成指标
SDj = abs(result_ - result_.mean()).sum()/result_.shape[0]
SCj = (result_ - result_.mean())/SDj
S = SCj.T.sum()
CI_currency = S


# In[27]:


plt.figure(figsize=(15,9),dpi = 750)
plt.plot(data)


# # 宏观-资产映射

# In[28]:


def backtest(df,time1,time2,name):
    for i in range(len(time1)):
        df1 = df[(df['date']>pd.to_datetime(time1[i][0])) & (df['date']<pd.to_datetime(time1[i][1]))]
        df2 = df[(df['date']>pd.to_datetime(time2[i][0])) & (df['date']<pd.to_datetime(time2[i][1]))]
        if i == 0:
            df_up,df_down = df1,df2
        elif i != 0:
            df_up,df_down = pd.concat([df_up,df1]),pd.concat([df_down,df2])           
    stock,bond,commodit,gold = [df_up['stock'].mean(),df_down['stock'].mean()],[df_up['bond'].mean(),df_down['bond'].mean()],[df_up['commodit'].mean(),df_down['commodit'].mean()],[df_up['gold'].mean(),df_down['gold'].mean()]
    df_result = pd.DataFrame([stock,bond,commodit,gold])
    df_result.index,df_result.columns = ['stock','bond','commodit','gold'],[name + '上行',name + '下行']
    return df_result


# In[29]:


macro_fac = pd.DataFrame([CI_increase,CI_inflation,CI_credit,CI_currency]).T
macro_fac.columns = ['increase','inflation','credit','currency']


# In[30]:


stock = w.wsd("881001.WI", "close", "2005-12-31", "2022-05-31", "Period=M;Days=Alldays",usedf = True)[1].pct_change(periods = 1).dropna()
stock.columns = ['stock']
bond = w.wsd("CBA00101.CS", "close", "2005-12-31", "2022-05-31", "Period=M;Days=Alldays",usedf = True)[1].pct_change(periods = 1).dropna()
bond.columns = ['bond']
commodit = w.wsd("NH0100.NHF", "close", "2005-12-31", "2022-05-31", "Period=M;Days=Alldays",usedf = True)[1].pct_change(periods = 1).dropna()
commodit.columns = ['commodit']
gold = w.wsd("AU9999.SGE", "close", "2005-12-31", "2022-05-31", "Period=M;Days=Alldays",usedf = True)[1].pct_change(periods = 1).dropna()
gold.columns = ['gold']


# In[31]:


df = macro_fac.join([stock,bond,commodit,gold]).dropna()
df['date'] = pd.to_datetime(df.index)


# In[32]:


increase_time1 = [('2009/1/31','2010/3/31'),('2012/4/30','2013/11/30'),('2015/7/31','2017/8/31'),('2020/2/29','2021/2/28')]
increase_time2 = [('2010/3/31','2012/4/30'),('2013/11/30','2015/7/31'),('2017/8/31','2020/2/29'),('2021/2/28','2022/2/28')]

inflation_time1 = [('2009/3/31','2011/7/31'),('2012/6/30','2013/9/30'),('2015/5/31','2017/2/28'),('2018/12/31','2020/1/31'),('2020/7/31','2021/4/30')]
inflation_time2 = [('2011/7/31','2012/6/30'),('2013/9/30','2015/5/31'),('2017/2/28','2018/12/31'),('2020/1/31','2020/7/31'),('2021/4/30','2022/3/31')]

credit_time1 = [('2009/1/31','2009/11/30'),('2012/1/31','2013/1/31'),('2015/3/31','2017/2/28'),('2020/1/31','2020/9/30')]
credit_time2 = [('2009/11/30','2012/1/31'),('2013/1/31','2015/3/31'),('2017/2/28','2020/1/31'),('2020/9/30','2022/1/31')]

currency_time1 = [('2011/7/31','2012/7/31'),('2013/12/31','2015/7/31'),('2018/1/31','2020/4/30'),('2020/12/31','2022/3/31')]
currency_time2 = [('2009/3/31','2011/7/31'),('2012/7/31','2013/12/31'),('2015/7/31','2018/1/31'),('2020/4/30','2020/12/31')]


# In[33]:


df1 = backtest(df,increase_time1,increase_time2,'增长')
df2 = backtest(df,inflation_time1,inflation_time2,'通胀')
df3 = backtest(df,credit_time1,credit_time2,'信用')
df4 = backtest(df,currency_time1,currency_time2,'货币')


# In[34]:


#不同经济环境下资产收益率情况
pd.concat([df1,df2,df3,df4],axis = 1)*100


# # 宏观点位预测

# In[35]:


#周期判断，用正比例函数拟合42个月的基钦周期
def f_fit(x,a,b):
    return a*np.sin((2*np.pi)*x/42 + b)

def f_show(x,p_fit):
    a,b = p_fit.tolist()
    return a*np.sin((2*np.pi)*np.array(x)/42 + b)

def period(alist):
    result = []
    for i in range(32,63):
        x = [j for j in range(i)]
        y = alist[len(alist)-i:]
        p_fit,pcov=curve_fit(f_fit,x,y)
        threshold = abs(p_fit[0]*np.sin(np.pi/3))
        if -threshold < f_show(i+1,p_fit) < threshold:
            if f_show(i+1,p_fit) > f_show(i,p_fit):
                result.append(1)
            elif f_show(i+1,p_fit) < f_show(i,p_fit):
                result.append(-1)

        if f_show(i+1,p_fit) >= threshold:
            if y[-1] > y[-2]:
                result.append(1)
            elif y[-1] < y[-2]:
                result.append(0)

        if f_show(i+1,p_fit) <= -threshold:
            if y[-1] < y[-2] :
                result.append(-1)
            elif y[-1] > y[-2]:
                result.append(0)
    return pd.DataFrame(result).mode().values[0][0]

#动量判断
def momentum(alist):
    if alist[-1] - alist[-2] > 0 and alist[-2] - alist[-3] > 0:
        result = 1
    elif alist[-1] - alist[-2] < 0 and alist[-2] - alist[-3] < 0:
        result = -1
    else:
        result = 0
    return result


# # 周期拟合效果

# In[36]:


def plot_period(alist):
    x = [j for j in range(len(alist))]
    y = alist
    p_fit,pcov=curve_fit(f_fit,x,y)
    plt.style.use('ggplot')
    plt.figure(figsize=(15,9),dpi = 750)
    plt.plot(f_show(x,p_fit))
    plt.plot(alist)


# In[37]:


plot_period(CI_increase.tolist())


# In[38]:


plot_period(CI_inflation.tolist())


# In[39]:


plot_period(CI_credit.tolist())


# In[40]:


plot_period(CI_currency.tolist())


# # 大类资产配置回测

# In[41]:


#根据风险预算模型计算不同风险占比下资产的配置权重
def cacu_weight(stock_r,bond_r,commodit_r,b_w):
    df1 = pd.DataFrame(stock_r['stock'].tolist()[-252:],columns = ['stock'])
    df2 = pd.DataFrame(bond_r['bond'].tolist()[-252:],columns = ['bond'])
    df3 = pd.DataFrame(commodit_r['commodit'].tolist()[-252:],columns = ['commodit'])
    df_ = pd.concat([df1,df2,df3],axis = 1)
    sigma_s = df_['stock'].std()*np.sqrt(252)
    sigma_b = df_['bond'].std()*np.sqrt(252)
    sigma_c = df_['commodit'].std()*np.sqrt(252)
    w0 = (sigma_b*b_w[0])/(sigma_s*b_w[1])
    w2 = (sigma_b*b_w[2])/(sigma_c*b_w[1])
    return [w0/(w0+1+w2),1/(w0+1+w2),w2/(w0+1+w2)]

#对下一期宏观环境进行预测（周期与动量两个条件相结合）
def forecast(macro_name):
    result = []
    for i in range(63,len(macro_index)):
        m = macro_index.iloc[:i,:]
        a = period(m[macro_name].tolist())
        b = momentum(m[macro_name].tolist())
        if a + b > 0:
            r = 1
        elif a + b == 0:
            r = 0
        elif a + b < 0:
            r = -1
        result.append(r)
    return result

#根据大类资产与宏观经济条件映射关系做出风险预算调整
def risk_change(w,alist,gamma):
    if alist[0] == 1:
        w = w*np.array([gamma,1,gamma])/sum(w*np.array([gamma,1,gamma]))
    elif alist[0] == -1:
        w = w*np.array([1/gamma,1,1/gamma])/sum(w*np.array([1/gamma,1,1/gamma]))
    
    if alist[1] == 1:
        w = w*np.array([1,1,gamma])/sum(w*np.array([1,1,gamma]))
    elif alist[1] == -1:
        w = w*np.array([1,1,1/gamma])/sum(w*np.array([1,1,1/gamma]))

    if alist[2] == 1:
        w = w*np.array([gamma,1,1])/sum(w*np.array([gamma,1,1]))
    elif alist[2] == -1:
        w = w*np.array([1/gamma,1,1])/sum(w*np.array([1/gamma,1,1]))

    if alist[3] == 1:
        w = w*np.array([1,gamma,1])/sum(w*np.array([1,gamma,1]))
    elif alist[3] == -1:
        w = w*np.array([1,1/gamma,1])/sum(w*np.array([1,1/gamma,1]))
    
    return w

#回测结果函数
def result_fun(return_list,rate_list):
    result = []
    i = np.argmax((np.maximum.accumulate(return_list)- return_list)/np.maximum.accumulate(return_list))
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])    
    MaxDrawdown = -(return_list[j] - return_list[i]) / return_list[j]
    return_year = pow(return_list[-1],12/len(return_list))-1
    volatility = np.std(np.array(rate_list) - 1)*pow(12,0.5)   
    result.append(return_list[-1])
    result.append(return_year*100)
    result.append(volatility*100)
    result.append(MaxDrawdown*100)
    result.append(result[1]/result[2])    
    result = pd.DataFrame(result).T
    result.columns = ['净值','年化收益率','年化波动率','最大回撤','夏普比率']    
    return result


# In[42]:


def get_epr(bar):
    bond_rate = w.edb('S0059749',"20041231","","Fill=Previous;Days=Alldays",usedf = True)[1]
    PE = w.wsd("881001.WI", "pe_ttm", "2004-12-31", "", "ruleType=10;Days=Alldays",usedf = True)[1]
    df_epr = 1/PE.join(bond_rate)
    df_epr['EPR'] = df_epr['PE_TTM'] - df_epr['CLOSE']
    rank_25,rank_75 = [],[]
    for i in range(365*3,len(df_epr)):
        rank_25.append(np.percentile(df_epr['EPR'].iloc[i-365*3:i], bar))
        rank_75.append(np.percentile(df_epr['EPR'].iloc[i-365*3:i], 100-bar))
    df_epr = pd.concat([df_epr['EPR'].iloc[365*3:],pd.DataFrame(rank_25,index = df_epr['EPR'].iloc[365*3:].index),pd.DataFrame(rank_75,index = df_epr['EPR'].iloc[365*3:].index)],axis = 1)
    df_epr.columns = ['EPR','rank_25','rank_75']
    return df_epr


# In[43]:


macro_index = df[['increase','inflation','credit','currency']]
market_index = df[['stock','bond','commodit']]
increase_r = forecast('increase')
inflation_r = forecast('inflation')
credit_r = forecast('credit')
currency_r = list(np.array(forecast('currency'))*(-1))
df_result = pd.concat([pd.DataFrame(increase_r),pd.DataFrame(inflation_r),pd.DataFrame(credit_r),pd.DataFrame(currency_r)],axis = 1)
df_result.columns = ['increase','inflation','credit','currency']
df_result.index = pd.date_range("2012-06-30",periods = len(df_result), freq="M")


# In[44]:


stock_r = w.wsd("881001.WI", "close", "2008-12-31", "2022-05-31", "Period=D",usedf = True)[1].pct_change(periods = 1).dropna()
stock_r.columns = ['stock']
bond_r = w.wsd("CBA00101.CS", "close", "2008-12-31", "2022-05-31", "Period=D",usedf = True)[1].pct_change(periods = 1).dropna()
bond_r.columns = ['bond']
commodit_r = w.wsd("NH0100.NHF", "close", "2008-12-31", "2022-05-31", "Period=D",usedf = True)[1].pct_change(periods = 1).dropna()
commodit_r.columns = ['commodit']
gold_r = w.wsd("AU9999.SGE", "close", "2008-12-31", "2022-05-31", "Period=D",usedf = True)[1].pct_change(periods = 1).dropna()
gold_r.columns = ['gold']
stock_r['date'] = pd.to_datetime(stock_r.index)
bond_r['date'] = pd.to_datetime(bond_r.index)
commodit_r['date'] = pd.to_datetime(commodit_r.index)
gold_r['date'] = pd.to_datetime(gold_r.index)


# In[45]:


df_backtest = df_result.join(market_index).dropna()
df_backtest['date'] = df_backtest.index
date_index = df_backtest.index


# In[46]:


date_index = date_index[36:]


# In[47]:


df_epr = get_epr(5)
df_epr['date'] = df_epr.index


# In[48]:


#回测
rate_me = []
rate_base = []
weight_all = []
risk_base = [0.8,0.2,0] #这里可以随意调整，然后看图像结果
for i in trange(len(date_index)):
    d = df_backtest[df_backtest['date'] == date_index[i]]
    epr = df_epr[df_epr['date'] == date_index[i]]
    if epr.EPR.tolist()[0] < epr.rank_25.tolist()[0]:
        b_w = np.array([0.5,1,1])*np.array(risk_base) / sum(np.array([0.5,1,1])*np.array(risk_base))
    else:
        b_w = risk_change(np.array(risk_base),d.iloc[:,:4].values[0].tolist(),2)
        
        
    stock_input = stock_r[stock_r['date'] < date_index[i]]
    bond_input = bond_r[bond_r['date'] < date_index[i]]
    commodit_input = commodit_r[commodit_r['date'] < date_index[i]]
    weight_me = cacu_weight(stock_input,bond_input,commodit_input,b_w)
    weight_base = cacu_weight(stock_input,bond_input,commodit_input,risk_base)
    
    rate_me.append(sum(np.array(weight_me)*(d.iloc[:,4:-1].values[0] + 1)))
    rate_base.append(sum(np.array(weight_base)*(d.iloc[:,4:-1].values[0] + 1)))
    weight_all.append(weight_me)


# In[49]:


weight_base


# In[50]:


value_me = pd.DataFrame(rate_me).cumprod()
value_base = pd.DataFrame(rate_base).cumprod()
value_me.index = date_index
value_base.index = date_index


# In[51]:


weight_all = pd.DataFrame(weight_all)
weight_all.index = date_index
weight_all.columns = ['stock','bond','commodit']


# In[52]:


plt.figure(figsize=(15,9),dpi = 750)
plt.plot(value_me,label = 'strategy')
plt.plot(value_base,label = 'base')
plt.legend()


# In[53]:


#回测结果
result_me = result_fun(value_me[0].tolist(),rate_me)
result_base = result_fun(value_base[0].tolist(),rate_base)

result_stock = result_fun((df_backtest['stock']+1).cumprod().tolist(),df_backtest['stock'].tolist())
result_bond = result_fun((df_backtest['bond']+1).cumprod().tolist(),df_backtest['bond'].tolist())
result_commodit = result_fun((df_backtest['commodit']+1).cumprod().tolist(),df_backtest['commodit'].tolist())

result_me.index,result_base.index = ['策略'],['基准']

result = pd.concat([result_me,result_base])
result


# In[54]:


#回测区间内策略每期的配置情况
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
weight_all


# In[ ]:




