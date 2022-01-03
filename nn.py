import numpy as np
import pygame as py
import sys,time,pickle

class net:
  network=[]

  def __init__(self,shape,seed=42, node_input=[]):
    "creates a 'newtork' consisting of a random seed for each value creating,the node array, weight array and bias array"
    "shape is a tuple object, seed is random number seed, node input is the initial input values"
    node_matrix = []
    w_arrays = []
    b_matrix = []
    
    for i in range(len(shape)):
      w_matrix = []
      y = shape[i]

      if len(node_input) == 0 and i==0:
        node_vector = np.random.default_rng(seed).random((y,1))  #insert random values node bias
      else:
        node_vector = node_input

      if i+1 < len(shape):
        for j in range(y):
          w_vector = np.random.default_rng(seed+1).random((shape[i+1],1))
          w_matrix.append(w_vector)

        w_matrix = np.array(w_matrix)
        w_arrays.append(w_matrix)

      if 0<i :
        b_vector = np.random.default_rng(seed+2).random((y,1))
        b_matrix.append(b_vector)
      
      node_matrix.append(node_vector)


    self.network = [node_matrix,
                    w_arrays,
                    b_matrix]

  def for_step(self,node_vector, w_matrix, b_vector):
    ## calculates the output of a single step along the network
    
    out_vector=[]
    for i in range(len(w_matrix[0,:])):
      weight = np.dot(np.array(node_vector).transpose(),
                      w_matrix[:,i])[0]
      total = (weight + b_vector[i])[0]
      result = self.sigmoid(total)
      out_vector.append(result)

    return out_vector

  def for_pass(self):
    ## completes a forward pass over the whole network
    self.f_pass_out=[]
    for i in range(len(self.network[0])-1):
      step = i
      if i==0:
        node_vector = self.network[0][step]
      else:
        node_vector = output
        
      w_matrix = self.network[1][step]
      b_vector = self.network[2][step]

      output = self.for_step(node_vector, w_matrix, b_vector)
      self.f_pass_out.append(output)

    f_pass = open("New_file.txt","wt")
    f_pass.write(str(self.f_pass_out))
    f_pass.close()

  def error_calc(self,actual,model,r_max,rotators):
    print("Error calc")
    idx,min_dist_list = self.closest_match(actual,model)
    self.error = np.sum(np.array(min_dist_list)**2)
    self.cost_list = []
    for i in range(rotators):
      cost_i = self.sigmoid(self.error*r_max/(10**9))
      self.cost_list.append(cost_i/(i+1))
    
    total_cost = (sum(self.cost_list)**2)/(2*rotators)
    print(self.cost_list,total_cost)
    self.error_vector = np.array(self.cost_list) 

  def error_calc_norm(self,actual,model):
    cost_sum = 0
    cost_vector=[]
    for i in range(len(actual)):
      cost = abs(actual[i] - model[i])/2
      cost_sum += cost
      cost_vector.append(cost)

    self.error_vector=np.array(cost_vector)

  def closest_match(self, x1, x2):
    print("Closest match")
    max_1 = len(x1)
    max_2 = len(x2)
    if max_1 != max_2:
      if max_1 > max_2:
        while len(x1) > max_2:
          x = np.random.uniform(0,len(x1)-1)
          x1 = np.delete(x1,int(x),0)
      elif max_1 < max_2:
        while len(x2) > max_1:
          x = np.random.uniform(0,len(x2)-1)
          x2 = np.delete(x2,int(x),0)
    
    index_array=[]
    min_dist_list=[]
    for i in range(len(x1)):
      min_dist=np.inf
      idx=0
      for j in range(len(x2)):
        dist = np.sqrt((x1[i][0]-x2[i][0])**2+(x1[i][1]-x2[i][1])**2)
        if dist < min_dist:
          idx = j
          min_dist = dist
      index_array.append(idx)
      min_dist_list.append(min_dist)
    return index_array,min_dist

  def sigmoid(self,value):
    ## squishification function to keep it within 0 -> 1
      y = 1/(1+np.exp(-value))
      return y
    
  def sigmoid_prime(self,value):
    result = np.exp(-value) / (1+np.exp(-value)**2)
    return result

  def back_step(self,mu,step):
    print("back step")
    z_l = self.f_pass_out[len(self.f_pass_out)-step]
    a_l = self.f_pass_out[len(self.f_pass_out)-step-1]
    a_p = self.network[0][0]

    w_l = self.network[1][len(self.network[1])-step]
    b_vector = self.network[2][len(self.network[2])-step]

    error_list = []
    w_error_matrix = []
    if step == 1:
      print("step 1")
      for i in range(len(b_vector)):
        factor = (a_l[i] - self.error_vector[i])
        error = self.sigmoid_prime(z_l[i]) * factor
        error_list.append(error)
        w_error_list = []
        for j in range(len(a_l)):
          w_error_list.append(a_l[j] * error)

        w_error_matrix.append(w_error_list)
    elif step == len(self.f_pass_out):
      print("Step :",step)
      for i in range(len(b_vector)):
        w_j = self.network[1][len(self.network[1])-step+1]
        if len(self.error_vector.shape) == 1:
          self.error_vector = self.error_vector.reshape(self.error_vector.shape[0],1)
        #print(w_j.transpose()[0].transpose().shape,self.error_vector.shape,"x1,x2")
        factor = np.matmul(w_j.transpose()[0].transpose(),self.error_vector)
        error = (self.sigmoid_prime(z_l[i]) * factor[0])[0]
        error_list.append(error)
        w_error_list = []
        for j in range(len(a_p)):
          w_error_list.append(a_p[j] * error)

        w_error_matrix.append(w_error_list)
    else:
      print("Step :",step)
      for i in range(len(b_vector)):
        w_j = self.network[1][len(self.network[1])-step+1]
        if len(self.error_vector.shape) == 1:
          self.error_vector = self.error_vector.reshape(self.error_vector.shape[0],1)
        #print(w_j.transpose()[0].transpose().shape,self.error_vector.shape,"x1,x2")
        factor = np.matmul(w_j.transpose()[0].transpose(),self.error_vector)
        error = (self.sigmoid_prime(z_l[i]) * factor[0])[0]
        error_list.append(error)
        w_error_list = []
        for j in range(len(a_l)):
          w_error_list.append(a_l[j] * error)

        w_error_matrix.append(w_error_list)
    
    self.error_vector = np.array(error_list)
    w_error_matrix = np.array(w_error_matrix)

    if len(w_error_matrix.shape) == 3:
      replacement = np.zeros(w_l.shape)
      for i in range(len(w_l)):
        for j in range(len(w_l[0])):
          x = np.array(w_error_matrix).transpose()
          replacement[i][j] = [w_l[i][j][0]+mu*x[0][i][j]]
    else:
      replacement = np.zeros(w_l.shape)
      for i in range(len(w_l)):
        for j in range(len(w_l[0])):
          x = np.array(w_error_matrix).transpose()
          replacement[i][j] = [w_l[i][j][0]+mu*x[i][j]]

    print("error :",error_list[0])
    new_w = replacement
    new_b = b_vector + mu*np.array(error_list)[0]
    return new_w, new_b
    
  def back_pass(self,mu):
    steps = len(self.network[0])-1
    for i in range(steps):
      w, b = self.back_step(mu,i+1)
      #print(self.network[1][len(self.network[1])-i-1].shape,w.shape)
      self.network[1][len(self.network[1])-i-1] = w
      #print(self.network[2][len(self.network[2])-i-1].shape,b.shape)
      self.network[2][len(self.network[2])-i-1] = b

  def save(self):
    myFile = open("network_Save.pkl","wb")
    pickle.dump(self.network,myFile)
    myFile.close()
       
class rotator:
  positions = []
  color=[]
  def __init__(self, centre, ang_vel, r, phase = 0, prev="null"):
    ## Defines a roator object with a centre, link to previous,
    ## the angular velocity and length of the vector produced.
    self.ang_vel = ang_vel
    self.prev = prev
    self.r = r
    self.phase = phase

    if self.prev == "null":
      self.centre = centre
      self.pos = (centre[0] + self.r * np.cos(float(phase)),
                  centre[1] + self.r * np.sin(float(phase)))
    else:
      self.pos = (self.prev.pos[0] + self.r*np.cos(phase),
                  self.prev.pos[1] + self.r*np.sin(phase))

  def update(self, time=0):
    ## Updates the position of each rotator recursively
    if self.prev == "null":
      x = np.cos(self.ang_vel * time /1000 + self.phase)
      y = np.sin(self.ang_vel * time /1000 + self.phase)
      self.pos = (self.centre[0] + self.r * x,
                  self.centre[1] + self.r * y)
    else:
      x = np.cos(self.ang_vel * time /1000 + self.phase)
      y = np.sin(self.ang_vel * time /1000 + self.phase)
      self.prev.update(time)
      self.pos = (self.r*x + self.prev.pos[0],
                  self.r*y + self.prev.pos[1])

  def draw(self,surface, time, const=False, width_l=2):
    ## Draws and records the positions created by the rotators
    
    old_pos = self.pos
    self.positions.append(old_pos)
    self.update(time)
    new_pos=self.pos

    E = np.sqrt((new_pos[0]-old_pos[0])**2+(new_pos[1]-old_pos[1])**2)
    surface.fill((0,0,0))
    thresh = 4
    if E >thresh:
      E=thresh
    self.color.append((255,120,150))
    
    if const == True:
      self.draw_circ(surface)
      self.draw_line(surface,width=width_l)

    for i in range (len(self.positions)):
      py.draw.circle(surface,
                     self.color[i],
                     self.positions[i],
                     2)

  def draw_circ(self,surface,thickness=2):
    ## Used to create the radius circles
    if self.prev != "null":
      self.prev.draw_circ(surface)
      py.draw.circle(surface,(255,255,255),self.prev.pos,self.r)
      py.draw.circle(surface,(0,0,0),self.prev.pos,self.r-thickness)
    else:
      py.draw.circle(surface,(255,255,255),self.centre,self.r)
      py.draw.circle(surface,(0,0,0),self.centre,self.r-thickness)

  def draw_line(self,surface,color=(20,200,255),width=2):
    ## Used to draw the lines along the point vector
    if self.prev == "null":
      py.draw.line(surface,color,self.centre,self.pos,width)
    else:
      py.draw.line(surface,color,self.prev.pos,self.pos,width)
      self.prev.draw_line(surface)

class generator(rotator):
  ## inherits the rotator class but used to create many either random
  ## or pre determined rotators with different ang velocities.
  obj_list=[]
  def __init__(self, rot_number, centre,r_max, phase=0, w_list= []):
    ## creates the rotator objects and relations
    for i in range(rot_number):
      if w_list == []:
        w = np.round(np.random.random(),2)
      else:
        w = w_list[i]
        
      if  0< w < 0.01:
        w=0.01
      r = r_max/(i+1)
      if i == 0:
        x = rotator(centre,w,r,phase)
      else:
        x = rotator(centre,w,r,phase, self.obj_list[i-1])
      self.obj_list.append(x)
    self.end = self.obj_list[len(self.obj_list)-1]



  def update(self,time=0):
    ## calls update using the last object linked in __init__
    self.end.update(time)

  def draw(self, surface, time, const=False,width=2):
    ## calls the draw function using the last linked object
    self.end.draw(surface,time,const,width)

class fourier_gen(generator):
  def __init__(self, centre,r_avg,amp,a_list, phase=0, w_list= []):
    ## creates the rotator objects and relations
    norm_val = np.sum(a_list)
    norm_const = 1/norm_val
    idx = a_list.index(min(a_list))
    for i in range(len(w_list)):
      w = w_list[i]

      if i == 0:
        w = w_list[idx]
        self.w0 = w
        x = rotator(centre,w,r_avg,phase)
      elif i == idx:
        x = rotator(centre,w,a_list[0][0]*amp*norm_const,phase, self.obj_list[i-1])
      else:
        x = rotator(centre,w,a_list[i][0]*amp*norm_const,phase, self.obj_list[i-1])
      self.obj_list.append(x)
    self.end = self.obj_list[len(self.obj_list)-1]

class mouse_shape:
  def __init__(self):
    self.pixels = []
    
  def update(self,size =3):
    on = py.mouse.get_pressed(num_buttons=3)[0]
    
    if on:
      pos = py.mouse.get_pos()
      x=pos[0]
      y=pos[1]
      for i in range(-(size-1),size):
        for j in range(-(size-1),size):
          pos = [x+i,y+j]
          self.pixels.append(pos)

  def draw(self,screen,size=1):
    screen.fill((0,0,0))
    self.update(size)
    for i in range(len(self.pixels)):
      py.draw.circle(screen,(255,255,255),self.pixels[i],1)
















