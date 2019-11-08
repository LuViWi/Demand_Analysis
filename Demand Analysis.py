#!/usr/bin/env python
# coding: utf-8

# ![grafik.png](attachment:grafik.png)

# ## imports

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


# ## helper funktions

# In[2]:


def ref_val(x):
    return x


# In[3]:


def get_ref_val(time_space, rel_dict, rating_dict, threshold=0, method="sold_out", top_n=100, plot=False):
    if method == "sold_out":
        n = 0
        Y = np.zeros_like(time_space)
        for key in list(rel_dict.keys()):
            if rel_dict[key]["sold_out"]:
                n += 1
                y = step_funktion(rel_dict[key]["dates"], rel_dict[key]["capacity"], time_space)
                Y += y
        Y = Y / n
    elif method == "threshold":
        if not rating_dict:
            Y = ref_val(time_space)
        else:
            n = 0
            Y = np.zeros_like(time_space)
            for key in list(rating_dict.keys()):
                if rating_dict[key]["weighted_score"] > threshold:
                    n += 1
                    y = step_funktion(rel_dict[key]["dates"], rel_dict[key]["capacity"], time_space)
                    Y += y
            Y = Y / n
    elif method == "hierarchie":
        if not rating_dict:
            Y = ref_val(time_space)
        else:
            hierarchielist = []
            for key in list(rating_dict.keys()):
                hierarchielist.append([key, rating_dict[key]["weighted_score"]])
            hierarchielist = sorted(hierarchielist, key=lambda a_entry: a_entry[1], reverse=True)
            Y = np.zeros_like(time_space)
            if top_n > len(hierarchielist):
                print("top_n too big. max:", len(hierarchielist), "set to:", int(len(hierarchielist) / 2))
                top_n = int(len(hierarchielist) / 2)
            for key in np.array(hierarchielist)[:top_n, 0]:
                y = step_funktion(rel_dict[key]["dates"], rel_dict[key]["capacity"], time_space)
                Y += y
                if plot:
                    plt.plot(rel_dict[key]["dates"], rel_dict[key]["capacity"], "--g", alpha=0.5)
                    plt.scatter(rel_dict[key]["dates"][-1], rel_dict[key]["capacity"][-1])
            Y = Y / top_n

    return Y


# In[32]:


def weight(x, n=2):
    return x ** n


# -np.exp(-100*x)+1


# In[33]:


def step_funktion(x, y, space):
    N = len(space)
    n = 0
    i = 0
    funktion = []
    while i < len(x):
        while n < N:
            if i == len(x) - 1:
                funktion.append(y[-1])
            else:
                if x[i] > space[n]:
                    if i == 0:
                        funktion.append(0)
                    else:
                        funktion.append(y[i - 1])
                else:
                    i += 1
                    break
            n += 1
            if n == N:
                i += 1
                break
    return funktion


# ## Parameters

# In[34]:


rating_dict = {}
n_samples = 10000
threshold = -0.5
time_space = np.linspace(0, 1, n_samples)
filter_cat = "Service Promoter Name"
filter_for = "MAWI Concert Konzertagentur GmbH"
method = "hierarchie"
top_n = 30

# # Load Data

# In[35]:


df = pd.read_csv("csv_files/Random Sample.csv", sep=";")

# In[36]:


keys = ["Event Name", "Event Id", "Sum Tickets Booked", "Total Seats 2",
        "Ticket Booking Date", "Event Date", "Event Onsale Date", "Artist Name", "Event Series Name", filter_cat]
df = df[keys]
vals = df.values

# # Parse Dates

# In[37]:


for val in vals:
    for n in [4, 5, 6]:
        if type(val[n]) == str:
            val[n] = datetime.datetime.strptime(val[n], '%d.%m.%Y %H:%M:%S')

# # dtype Check

# In[38]:


print("---")
for v in vals[0]:
    print(v)
    print(type(v))
    print("---")

# # Save Events + features to Booking dict

# In[39]:


booking_dict = {}
for val in vals:
    if True:  # val[-1] == filter_for:
        val[0] = val[0] + "_id_" + str(val[1])
        # val[0]=str(val[1])
        if not val[0] in booking_dict.keys():
            booking_dict[val[0]] = {
                "max_capacity": val[3],
                "onsale_date": val[6],
                "event_date": val[5],
                "presale": (type(val[6]) == float),
                "booked_tickets": [],
                "booking_date": [],
                "valid": True
            }
        if booking_dict[val[0]]["max_capacity"] == val[3] and booking_dict[val[0]]["event_date"] == val[5] and \
                booking_dict[val[0]]["presale"] == (type(val[6]) == float):
            if not (type(val[6]) == float):
                if booking_dict[val[0]]["onsale_date"] == val[6]:
                    booking_dict[val[0]]["booked_tickets"].append(val[2])
                    booking_dict[val[0]]["booking_date"].append(val[4])

        elif booking_dict[val[0]]["valid"]:
            print("error! @", val[0])
            print("max_capacity", booking_dict[val[0]]["max_capacity"], val[3])
            print("onsale_date", booking_dict[val[0]]["onsale_date"], val[6])
            print("event_date", booking_dict[val[0]]["event_date"], val[5])
            print("presale", booking_dict[val[0]]["presale"], (type(val[6]) == float))
            print("not Valid!\n")
            booking_dict[val[0]]["valid"] = False

        if booking_dict[val[0]]["presale"]:
            booking_dict[val[0]]["valid"] = False

# # commentar
#
# error! @ 102 Boyz - Asozial Allstars Tour - Extended Edition
# max_capacity 294 500
# onsale_date 2019-04-06 13:00:00 2018-12-13 13:00:00
# event_date 2019-11-28 20:00:00 2019-11-27 20:00:00
# presale False False
# not Valid!
#
# Recherche ergab bei der show am 2019-11-28 handelt es sich um eine zusatzshow, ist nicht als 1 event wertbar, könnte sich be jedem herausgefiltertem event so verhalten. daher wird filterung beibehalten
#
# # commentar ende

# # Calculate normalized/relative Values
# negative rel_dates and capacities over 100% will be filtered out

# In[40]:


rel_dict = {}
for key in list(booking_dict.keys()):
    if booking_dict[key]["valid"]:
        # 1. sort for time
        M = np.vstack((booking_dict[key]["booking_date"], booking_dict[key]["booked_tickets"])).T

        M = M[M[:, 0].argsort()]

        # 2. get running total
        for i in range(len(M[:, 1])):
            if not i == 0:
                M[i, 1] = M[i, 1] + M[i - 1, 1]

        # 3. compute relative date
        onsale_date = booking_dict[key]["onsale_date"]
        event_date = booking_dict[key]["event_date"]
        booking_dates = M[:, 0]

        span = (event_date - onsale_date).days + (event_date - onsale_date).seconds / 86400

        for row in range(len(booking_dates)):
            booking_dates[row] = ((booking_dates[row] - onsale_date).days +
                                  (booking_dates[row] - onsale_date).seconds / 86400)

        rel_dates = booking_dates / span

        # 4. compute relative capcity
        running_total_tickets = M[:, 1]
        rel_capacity = running_total_tickets / booking_dict[key]["max_capacity"]

        if max(rel_capacity) > 1 or not (len(rel_capacity) == len(rel_dates)) or np.min(rel_dates) < 0:
            print(key, ": Not Valid!")
            booking_dict[key]["valid"] = False
        else:
            rel_dict[key] = {
                "dates": rel_dates,
                "capacity": rel_capacity,
                "n_rows": len(rel_dates),
                "sold_out": max(rel_capacity) == 1
            }

# # Plot rel_capacity over rel_dates
# All Events in valid intervall

# In[41]:


# for key in list(rel_dict.keys()):
#     plt.plot(rel_dict[key]["dates"],rel_dict[key]["capacity"])
# plt.plot(np.linspace(0,1,100),np.linspace(0,1,100))


# ## all sold out

# In[42]:


# n =0
# for key in list(rel_dict.keys()):
#     if rel_dict[key]["sold_out"]:
#         n+=1
#         plt.plot(rel_dict[key]["dates"],rel_dict[key]["capacity"])
#         print(key, max(rel_dict[key]["capacity"]) , min(rel_dict[key]["dates"]))
# plt.plot(np.linspace(0,1,100),np.linspace(0,1,100))
# print(n)


# In[84]:


# plt.plot(np.linspace(0,1,100),np.linspace(0,1,100))
#

# # Whats Hot Rating

# ## Calculation of the ratings

# In[43]:


y_ref = get_ref_val(time_space, rel_dict, rating_dict, threshold, method, top_n)  # ML
# y_ref = ref_val(time_space)

err_normalizing_val = np.mean((np.ones(n_samples) - y_ref))

pos_weighted_err_normalizing_val = np.mean(weight(time_space) * (np.ones(n_samples) - y_ref))
neg_weighted_err_normalizing_val = np.mean(weight(time_space) * y_ref)

for key in list(rel_dict.keys()):
    sum_err = 0
    weighted_sum_err = 0
    err_list = []
    weighted_err_list = []
    x = rel_dict[key]["dates"]
    y = step_funktion(x, rel_dict[key]["capacity"], time_space)
    errs = np.array(y) - np.array(y_ref)

    w_err = np.mean(errs * weight(time_space))
    if w_err > 0:
        weighted_err_normalizing_val = pos_weighted_err_normalizing_val
    else:
        weighted_err_normalizing_val = neg_weighted_err_normalizing_val
    rating_dict[key] = {
        "basic_score": np.mean(errs) / err_normalizing_val,  # /(rel_dict[key]["n_rows"]/alpha),
        "weighted_score": w_err / weighted_err_normalizing_val  # /(rel_dict[key]["n_rows"]/alpha)
    }

# ## Plot Events with score > threshold

# In[83]:


# for key in list(rating_dict.keys()):
#     if True: #rating_dict[key]["weighted_score"]>threshold:# and rating_dict[key]["weighted_score"]<0:
#         fig = plt.figure()
#         y = step_funktion(rel_dict[key]["dates"],rel_dict[key]["capacity"],time_space)
#         #plt.plot(time_space,y,label=key+str(rating_dict[key]))
#         plt.plot(time_space,y,label="\nBasic_Score:"+str(rating_dict[key]["basic_score"])+
#                  "\nWeighted_Score:"+str(rating_dict[key]["weighted_score"]))
#         plt.plot(time_space,y_ref,label="reference_funktion")
#         plt.plot(time_space,weight(time_space),label="weight_funktion")
#         plt.legend()


# In[45]:


method = "hierarchie"


# In[31]:


# for n in [0,0.1,0.3,0.5,1,2,7,20,100,1000]:#np.linspace(0,10,20):
#     plt.plot(time_space,time_space**n,label= "n="+str(n))
# plt.legend()


# In[62]:


def get_n(method, rating_dict, rel_dict):
    if method == "sold_out":
        n = 0
        for key in list(rating_dict.keys()):
            if rel_dict[key]["sold_out"]:
                n += 1
        print(n, "von", len(rating_dict.keys()), "Events sind ausverkauft")
    elif method == "threshold":
        n = 0
        for key in list(rating_dict.keys()):
            if rating_dict[key]["weighted_score"] >= threshold:
                n += 1
        print(n, "von", len(rating_dict.keys()), "werten liegen über dem Threshold von", threshold)
    elif method == "hierarchie":
        n = top_n
        print(top_n, " beste Curven nach weighted_score werden Berücksichtigt")
    return n


# # Improving with Curve fitting
# ## Calculate Average reference Curve of the events with a score > threshold

# In[85]:


# print(top_n)
top_n = 10
threshold = -0.5
method = "hierarchie"

# In[86]:


Y = get_ref_val(time_space, rel_dict, rating_dict, threshold, method, top_n)
print("Methode:", method)
print("Angepasste Kurve, mittelwert von", get_n(method, rating_dict, rel_dict), "Werten")
plt.plot(time_space, Y, lw=3)
plt.plot(time_space, time_space)

# In[87]:


# change_list= []
# for top_n in range(100):
#     Y = get_ref_val(time_space,rel_dict,rating_dict,threshold, method, top_n, plot = False)
#
#     change = np.mean(Y-time_space)
#     change_list.append(change)
#
#     plt.plot(time_space,Y,lw = 0.5)
#
#
# fig=plt.figure()
# fig.suptitle("change list", fontsize=10)
# plt.plot([change_list[i+1]-change_list[i] for i in range(len(change_list)-1)])


# ## Now Calculate Scores referencing new curve
# ### print both scores in comparison

# In[88]:


# y_ref = get_ref_val(time_space, rel_dict, rating_dict, threshold, method) #ML
# #y_ref = ref_val(time_space)

# err_normalizing_val = np.mean((np.ones(n_samples)-y_ref))
# weighted_err_normalizing_val = np.mean(weight(time_space)*(np.ones(n_samples)-y_ref))
# for key in list(rel_dict.keys()):
#     sum_err = 0
#     weighted_sum_err = 0
#     err_list = []
#     weighted_err_list =[]
#     x = rel_dict[key]["dates"]
#     y = step_funktion(x,rel_dict[key]["capacity"],time_space)
#     errs = np.array(y)-np.array(y_ref)
#     print(key)
#     print("basic:",rating_dict[key]["basic_score"],np.mean(errs)/err_normalizing_val)
#     print("weighted",rating_dict[key]["weighted_score"],np.mean(weight(time_space)*errs)/weighted_err_normalizing_val)
#     print(100*"-")
#     rating_dict[key] = {
#         "basic_score": np.mean(errs)/err_normalizing_val,#/(rel_dict[key]["n_rows"]/alpha),
#         "weighted_score":np.mean(weight(time_space)*errs)/weighted_err_normalizing_val#/(rel_dict[key]["n_rows"]/alpha)
#     }


# ## Plot Event Curves with reference to fitted Curve and new scores

# ### score > threshold

# In[23]:


# for key in list(rating_dict.keys()):
#     if rating_dict[key]["weighted_score"]>threshold:# and rating_dict[key]["weighted_score"]<0:
#         fig = plt.figure()
#         y = step_funktion(rel_dict[key]["dates"],rel_dict[key]["capacity"],time_space)
#         plt.scatter(time_space,y,label=key+str(rating_dict[key]), c = np.ones_like(y)+y-y_ref, s = 0.5)
#         plt.plot(time_space,y_ref,label="ref_val")
#         plt.plot(time_space,weight(time_space),label="weight_funk")
#         plt.legend()


# ### first 5

# In[24]:


# for key in list(rating_dict.keys())[:5]:
#     fig = plt.figure()
#     y = step_funktion(rel_dict[key]["dates"],rel_dict[key]["capacity"],time_space)
#     plt.scatter(time_space,y,label=key+str(rating_dict[key]), c = np.ones_like(y)+y-y_ref, s = 0.5)
#     plt.plot(time_space,y_ref,label="ref_funk")
#     plt.plot(time_space,weight(time_space),label="weight_funk")
#     plt.legend()


# # Export as csv

# In[25]:


out_dict = {
    "Event Id": [],
    "basic_score": [],
    "weighted_score": []
}

for key in list(rating_dict.keys()):
    out_dict["Event Id"].append(str(key))
    out_dict["basic_score"].append(rating_dict[key]["basic_score"])
    out_dict["weighted_score"].append(rating_dict[key]["weighted_score"])
if int(input("wanna save as csv? type 1 for yes, 0 for no.\nchoice:")) == 1:
    pd.DataFrame(out_dict).to_csv("csv_files/Eventim_scores.csv")
    print("csv saved!")
else:
    print("not saved!")

# # Forecast based on p % selling phase

# In[119]:


import random

p = 0.1
forecast_dict = {}
n_plots = 0
print("forecast auf basis der letzten ", 100 * p, "% der Verkaufsphase")
for key in list(rel_dict.keys()):  # [list(rel_dict.keys())[random.randint(0,len(rel_dict.keys()))]]:
    rel_dates = rel_dict[key]["dates"]
    rel_capacities = rel_dict[key]["capacity"]
    if len(rel_dates) >= 2:
        max_rel_date = rel_dates[-1]
        max_rel_capacity = rel_capacities[-1]
        if max_rel_date > p:
            arg = np.argwhere(rel_dates < (max_rel_date - p))
            if len(arg) > 1:
                arg = arg[-1][0]
                time_diff = max_rel_date - rel_dates[arg]
                val_diff = max_rel_capacity - rel_capacities[arg]
                forecast = ((1 - max_rel_date) * (val_diff / p) + max_rel_capacity)
                if forecast >= 1:
                    msg = "positive"
                else:
                    msg = "negative"

                sold_out_time = ((1 - max_rel_capacity) * (p / val_diff) + max_rel_date)
                forecast_dict[key] = {
                    "sold_out": True if msg == "positive" else False,
                    "score": 1 - sold_out_time if msg == "positive" else forecast - 1,
                    "p": p
                }

                # if max_rel_date<0.7:
                #     n_plots+=1
                #
                #     title =str(min(1,forecast)*100)+"% Auslastung\nScore"+str(forecast_dict[key]["score"])
                #
                #     if n_plots >2:
                #         plt.legend()
                #         fig = plt.figure()
                #         fig.suptitle(title, fontsize=10)
                #         n_plots =0
                #     plt.plot([0,1],[1,1],"--k")
                #     plt.plot(rel_dates,rel_capacities)
                #     plt.plot([max_rel_date-p,max_rel_date,min(sold_out_time,1)],[rel_capacities[arg],max_rel_capacity,min(1,forecast)],label=title)
                #     plt.scatter([max_rel_date-p,max_rel_date,min(sold_out_time,1)],[rel_capacities[arg],max_rel_capacity,min(1,forecast)])
                #

# # Forecast mittels Polynomal Regression

# In[27]:


import tensorflow as tf


def poly_reg_fit(x, y, poly_degree):
    n_rows = len(x)
    g_fit = tf.Graph()
    gpu_device_name = "/gpu:0"
    with g_fit.as_default():
        tf_x = tf.placeholder(shape=(n_rows, 1),
                              dtype=tf.float32,
                              name="tf_x")
        tf_y = tf.placeholder(shape=(n_rows, 1),
                              dtype=tf.float32,
                              name="tf_y")
        tf_X = tf.Variable(tf.ones(shape=(n_rows, 1),
                                   dtype=tf.float32),
                           name="tf_X")

        with tf.device(gpu_device_name):
            for deg in range(poly_degree):
                tf_X = tf.concat([tf_X, tf_x ** (deg + 1)], axis=1)

            w = tf.tensordot(tf.matmul(tf.linalg.inv(tf.matmul(tf.transpose(tf_X), tf_X)), tf.transpose(tf_X)), tf_y,
                             axes=1)

    with tf.Session(graph=g_fit) as sess:
        sess.run(tf.global_variables_initializer())
        weights = sess.run(w,
                           feed_dict={
                               tf_x: x.reshape(n_rows, 1),
                               tf_y: y.reshape(n_rows, 1)
                           })
    return weights.flatten()


# In[28]:


def poly_reg_pred(x, weights, poly_degree):
    n_rows = len(x)
    g_pred = tf.Graph()
    gpu_device_name = "/gpu:0"
    with g_pred.as_default():
        tf_x = tf.placeholder(shape=(n_rows, 1),
                              dtype=tf.float32,
                              name="tf_x")

        tf_weights = tf.placeholder(shape=(poly_degree + 1, 1),
                                    dtype=tf.float32,
                                    name="tf_weights")

        tf_X = tf.Variable(tf.ones(shape=(n_rows, 1),
                                   dtype=tf.float32),
                           name="tf_X")

        with tf.device(gpu_device_name):
            for deg in range(poly_degree):
                tf_X = tf.concat([tf_X, tf_x ** (deg + 1)], axis=1)

            tf_y = tf.tensordot(tf_X, tf_weights, axes=1)

    with tf.Session(graph=g_pred) as sess:
        sess.run(tf.global_variables_initializer())
        pred = sess.run(tf_y,
                        feed_dict=
                        {
                            tf_x: x.reshape(n_rows, 1),
                            tf_weights: weights.reshape(poly_degree + 1, 1)
                        }
                        )[:, 0]
    return pred


# In[112]:


poly_degree = 3
for key in [list(rel_dict.keys())[random.randint(0, len(rel_dict.keys()))] for i in range(3)]:
    rel_dates = rel_dict[key]["dates"]
    rel_capacities = rel_dict[key]["capacity"]

    weights = poly_reg_fit(rel_dates, rel_capacities, poly_degree)
    y = poly_reg_pred(time_space, weights, poly_degree)
    label = str(len(rel_dates)) + " Datenpunkte, Score: " + str(y[-1])
    plt.scatter(rel_dates, rel_capacities)
    plt.plot([0, 1], [1, 1], "--k")
    plt.plot(time_space, y, label=label)
plt.legend()

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:





