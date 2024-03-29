import numpy as np
import random
import matplotlib.pyplot as plt

# determine if two balls are touching
def setup_edges(center_x, center_y, r):
    edges = np.zeros((len(center_x), len(center_x)))

    # If ball n and ball m are touching, edges(n, m) = 1 only when n < m, else edges(n, m) = 0.
    # diagonal entries are automatically 0, as a ball is not allowed to collide with itself.
    for i in range(len(center_x)):
        for j in range(i, len(center_x)):
            x_i = np.array([center_x[i], center_y[i]])
            x_j = np.array([center_x[j], center_y[j]])
            if i != j:
                # allow an error of 1e-15 to accomodate for numerical error
                if np.linalg.norm(x_i-x_j) > (2*r - 1e-15) and np.linalg.norm(x_i-x_j) < (2*r + 1e-15):
                    edges[i, j] = 1
#                 elif np.linalg.norm(x_i-x_j) < (2*r - 1e-15):
#                     print("Overlapping detected. Double check input.")
    return edges

# check if ball i and ball j are allowed to have collisions. 
def check_collision(i, j, edges, center_x, center_y, v_x, v_y):
    if edges[i, j] != 1:
        return 0
    else:
        x_i = np.array([center_x[i], center_y[i]])
        x_j = np.array([center_x[j], center_y[j]])
        v_i = np.array([v_x[i], v_y[i]])
        v_j = np.array([v_x[j], v_y[j]])
        
        if np.dot(v_i-v_j, x_i-x_j) >= 0:
            return 0
        else:
            return 1

# if i, j collide, this method will assign them new velocity
def v_change(i, j, edges, center_x, center_y, v_x, v_y):
    x_i = np.array([center_x[i], center_y[i]])
    x_j = np.array([center_x[j], center_y[j]])
    u = (x_i - x_j)/np.linalg.norm(x_i - x_j)
    v_i = np.array([v_x[i], v_y[i]])
    v_j = np.array([v_x[j], v_y[j]])
    w_i = v_i + np.dot(v_j, u)*u - np.dot(v_i, u)*u
    w_j = v_j + np.dot(v_i, u)*u - np.dot(v_j, u)*u
    return w_i, w_j

# compute all collisions that can happen at a point based on the updated velocity
def all_collision(edges, center_x, center_y, v_x, v_y, r):
    contact = np.where(edges == 1)
    collide = []
    for i in range(len(contact[0])):
        if check_collision(contact[0][i], contact[1][i], edges, center_x, center_y, v_x, v_y) == 1:
            collide.append([contact[0][i], contact[1][i]])
    return collide

def perform_collision(i, j, edges, center_x, center_y, v_x, v_y):
    w_i, w_j = v_change(i, j, edges, center_x, center_y, v_x, v_y)    
    new_x = np.copy(v_x).astype('float64')
    new_y = np.copy(v_y).astype('float64')
    new_x[i] = w_i[0]
    new_y[i] = w_i[1]
    new_x[j] = w_j[0]
    new_y[j] = w_j[1]
    return new_x, new_y
    
# Perform a run that does all collisions until no more balls collide
def random_run(center_x, center_y, v_x, v_y, r, edges):
    collide = all_collision(edges, center_x, center_y, v_x, v_y, r)
#     [0, 0] denote the initial state
    step = [[0, 0]]

    this_x = np.copy(v_x).astype('float64')
    this_y = np.copy(v_y).astype('float64')
    v_xs = [this_x]
    v_ys = [this_y]
    
    indic = 0
    
    while len(collide) > 0:
        indic += 1
#         print(indic)
#         print("collide", collide)
        this_collide = random.randint(0, len(collide)-1)

        i = collide[this_collide][0]
        j = collide[this_collide][1]
        new_x, new_y = perform_collision(i, j, edges, center_x, center_y, this_x, this_y)
#         print(np.linalg.norm(new_x - this_x))
#         print(np.linalg.norm(new_y - this_y))
        to_remove = []
        if ((new_x - this_x).all() < 1e-15) and ((new_y - this_y).all() < 1e-15):
            to_remove.append([i, j])
#         print(new_x - this_x)
#         print(new_y - this_y)
            
#         print("to_remove", to_remove)
        this_x = new_x
        this_y = new_y

        collide = all_collision(edges, center_x, center_y, this_x, this_y, r)
#         print("next collide", collide)
        for i in np.arange(len(to_remove)):
            if to_remove[i] in collide: collide.remove(to_remove[i])
        step.append([i, j])
        v_xs.append(this_x)
        v_ys.append(this_y)
            
    return step, v_xs, v_ys

def visualize(center_x, center_y, v_x, v_y, r, ax):
    for i in range(len(center_x)):
        origin = (center_x[i], center_y[i])
        v = (v_x[i], v_y[i])
        circle = plt.Circle(origin, r, fill = False)
        label = ax.annotate(i, xy = origin, fontsize=8)
        if v_x[i] != 0 or v_y[i] != 0:
            ax.arrow(center_x[i], center_y[i], v_x[i], v_y[i], head_width = 0.2)
        ax.add_patch(circle)
        
    ax.set_xlim([min(center_x) - 2, max(center_x) + 2])
    ax.set_ylim([min(center_y) - 2, max(center_y) + 2])
    ax.set_aspect('equal')
    
def pinned_random_model(edges, center_x, center_y, v_x, v_y, r, location = "", save = False):
    step, v_xs, v_ys = random_run(center_x, center_y, v_x, v_y, r, edges)
    if save:
        fig, ax = plt.subplots(1, 1, figsize = (20, 20))
        ax.axis('off')
        for i in range(len(step)):
            visualize(center_x, center_y, v_xs[i], v_ys[i], r, ax)
            plt.savefig(location + "/" + str(i) + ".png")
            plt.cla()
    return len(step) - 1