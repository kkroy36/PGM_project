import random
from math import exp,sqrt

def get_int(l):

    return [float(i) for i in l]

def print_matrix(M):

    for item in M:
        print (item)


        
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

def get_dictionary(dat):

    data_dict = {}
    for item in dat:
        key = (item[0],item[1])
        value = item[2]
        data_dict[key] = value
    return (data_dict)

def dot_product(v1,v2):

    N = len(v1)
    return (sum([v1[i]*v2[i] for i in range(N)]))

def add_vectors(v1,v2):

    N = len(v1)
    return ([v1[i]+v2[i] for i in range(N)])

def sigmoid(x):

    return (exp(x)/float(1+exp(x)))

def compute_U_gradient(U,V,dat,gamma,d=10):

    M = len(V)
    N = len(U)
    data_dict = get_dictionary(dat)
    grad = []
    lambda_u = 0.002
    for i in [key[0] for key in data_dict]:
        total = [0 for i in range(d)]
        for j in [key[1] for key in data_dict]:
            if (i,j) in data_dict:
                R_ij = data_dict[(i,j)]
                u_i_dot_v_j = dot_product(U[i],V[j])
                diff = R_ij - u_i_dot_v_j
                first_component = [diff*item for item in V[j]]
                #second_component = [lamda_u*item for item in U[i]]
                #total_component = add_vectors(first_component,second_component)
                total = add_vectors(total,first_component)
        Ui_grad = [gamma*item for item in total]
        grad.append(Ui_grad)
    return (grad)

def compute_V_gradient(U,V,dat,gamma,d=10):

    M = len(V)
    N = len(U)
    data_dict = get_dictionary(dat)
    grad = []
    lambda_v = 0.002
    for i in [key[1] for key in data_dict]:
        total = [0 for i in range(d)]
        for j in [key[0] for key in data_dict]:
            if (j,i) in data_dict:
                R_ij = data_dict[(j,i)]
                u_i_dot_v_j = dot_product(U[j],V[i])
                diff = R_ij - u_i_dot_v_j
                first_component = [diff*item for item in U[j]]
                #second_component = [lamda_v*item for item in U[j]]
                #total_component = add_vectors(first_component,second_component)
                total = add_vectors(total,first_component)
        Vi_grad = [gamma*item for item in total]
        grad.append(Vi_grad)
    return (grad)
            
def add_matrices(m1,m2):

    total = []
    N = len(m1)
    for i in range(N):
        total.append(add_vectors(m1[i],m2[i]))
    return (total)

def transpose(m):

    m_t = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    return (m_t)
            

def PMF(dat,its=1000,gamma=0.01,d=10):

    inits = init_UV(dat,d=d)

    U = inits[0]
    V = inits[1]
    uid = inits[2]
    mid = inits[3]
    for it in range(its):
        gradient_U = compute_U_gradient(U,V,dat,gamma,d=d)
        gradient_V = compute_V_gradient(U,V,dat,gamma,d=d)
        U = add_matrices(U,gradient_U)
        V = add_matrices(V,gradient_V)
    return (U,V,uid,mid)

def product(X,Y):

    result = [[0 for j in range(len(Y[0]))] for i in range(len(X))]

    for i in range(len(X)):
        for j in range(len(Y[0])):
            for k in range(len(Y)):
               result[i][j] += X[i][k] * Y[k][j]

    return (result)

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

    pmf = PMF(data[:10],1000,0.005,d=d)
    U = pmf[0]
    V = pmf[1]
    uid = pmf[2]
    mid = pmf[3]
    V_t = transpose(V)
    R = product(U,V_t)
    data_dict = get_dictionary(data[:10])
    RMSE = compute_RMSE(R,data_dict,uid,mid)
    print (RMSE)

main()
