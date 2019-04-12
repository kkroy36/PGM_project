import random
from math import exp,sqrt
import numpy

def get_int(l):

    return [float(i) for i in l]

def dot_product(v1,v2):

    N = len(v1)
    return (sum([v1[i]*v2[i] for i in range(N)]))

def transpose(m):

    m_t = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    return (m_t)

def product(X,Y):

    result = [[0 for j in range(len(Y[0]))] for i in range(len(X))]
    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
               result[i][j] += X[i][k] * Y[k][j]
    return (result)

def get_dictionary(dat):

    data_dict = {}
    for item in dat:
        key = (item[0],item[1])
        value = item[2]
        data_dict[key] = value
    return (data_dict)

def init_UV(data,d=10):
    
    uid,mid = {},{}
    user_ids = [item[0] for item in data]
    unique_user_ids = list(set(user_ids))
    n_users = len(unique_user_ids)
    movie_ids = [item[1] for item in data]
    unique_movie_ids = list(set(movie_ids))
    n_movies = len(unique_movie_ids)
    U,V = [],[]
    for i in range(n_users):
        U.append([random.random() for k in range(d)])
        uid[unique_user_ids[i]] = i
    for j in range(n_movies):
        V.append([random.random() for k in range(d)])
        mid[unique_movie_ids[j]] = j
    return (U,V,uid,mid)

def compute_V_var(V):
    
    variance_matrix = []
    cols = len(V[0])
    for item in V:
        var = [[0 for i in range(cols)] for i in range(cols)]
        for i in range(cols):
            var[i][i] = 0.001
        variance_matrix.append(var)
    return (variance_matrix)

def compute_U_var(U):
    
    variance_matrix = []
    cols = len(U[0])
    for item in U:
        var = [[0 for i in range(cols)] for i in range(cols)]
        for i in range(cols):
            var[i][i] = 0.001
        variance_matrix.append(var)
    return (variance_matrix)

def compute_r_var(dat):

    data_dict = get_dictionary(dat)
    r_var = numpy.var(list(data_dict.values()))
    return (r_var)
        
def sample_U(U,V,dat,d=10):

    V_mean = [0 for i in range(d)]
    V_var = compute_V_var(V)
    R_var = compute_r_var(dat)
    for i in range(len(U)):
        V_j = None
        for j in range(len(V)):
            V_j = numpy.random.multivariate_normal(V_mean,V_var[j])
        u_i_dot_v_j = dot_product(U[i],V_j)
        R_i_j = numpy.random.normal(u_i_dot_v_j,R_var)
        U[i] = [R_i_j*V_j[k] for k in range(d)]
    return (U)

def sample_V(U,V,dat,d=10):

    U_mean = [0 for i in range(d)]
    U_var = compute_U_var(U)
    R_var = compute_r_var(dat)
    for j in range(len(V)):
        U_i = None
        for i in range(len(U)):
            U_i = numpy.random.multivariate_normal(U_mean,U_var[i])
        u_i_dot_v_j = dot_product(V[j],U_i)
        R_i_j = numpy.random.normal(u_i_dot_v_j,R_var)
        V[j] = [R_i_j*U_i[k] for k in range(d)]
    return (V)

def BPMF(dat,its=1000,d=10):

    inits = init_UV(dat,d=d)
    U = inits[0]
    V = inits[1]
    uid = inits[2]
    mid = inits[3]
    for it in range(its):
        print (it)
        U = sample_U(U,V,dat,d=d)
        V = sample_V(U,V,dat,d=d)
    return (U,V,uid,mid)

def compute_RMSE(R,data_dict,uid,mid):
    
    total_squared_error = 0
    for key in data_dict:
        value = data_dict[key]
        i = uid[key[0]]
        j = mid[key[1]]
        predicted_value = R[i][j]
        squared_error = (value - predicted_value)**2
        total_squared_error += squared_error
    mean_squared_error = total_squared_error/float(len(data_dict))
    return (sqrt(mean_squared_error))

def main():

    data = []
    d = 10
    with open("u.data","r") as fp:
        data = fp.read().splitlines()
        data = [item.split("\t") for item in data]
        data = [item[:-1] for item in data]
        data = [get_int(item) for item in data]

    bpmf = BPMF(data[:10],1000,d=d)
    U = bpmf[0]
    V = bpmf[1]
    uid = bpmf[2]
    mid = bpmf[3]
    V_t = transpose(V)
    R = product(U,V_t)
    data_dict = get_dictionary(data[:10])
    RMSE = compute_RMSE(R,data_dict,uid,mid)
    print (RMSE)

main()
