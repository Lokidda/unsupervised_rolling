import numpy as np

x = numpy.array(object, dtype = None)

#object = list/tuple...
#dtype: bool_, int8/16/32/64, uint, float16/32/64, complex64/128       ####要加'' 

#結構化數據類型
student = np.dtype([('name','S20'), ('age', 'i1'), ('marks', 'f4')]) 
a = np.array([('abc', 21, 50),('xyz', 18, 75)], dtype = student) 

#a['name'] == np.array(['abc','xyz'])   
#True

x.ndim
x.shape
x.size
x.dytpe
x.itemsize             #np對象中每個元素的大小
x.real
x.imag

np.empty(shape,dtype)
np.zeros(shape,dtype)
np.ones(shape,dtype)
np.eye
np.arange(start,end,step,dytype)
np.linspace(start,stop,num)
np.logspace(start,stop,num,base=10)
random.randn(shape)    #正態分佈
random.randint(low,high,(shape))

a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
a[...,1] == [2 4 5]
a[1,...] == [3 4 5]
a[...,1:] == [[2 3],[4 5],[5 6]]

a[a>3] == array([4, 5, 4, 5, 6])
a[#any condition that return bool]
a[[2,1]] == array([[4, 5, 6],[3, 4, 5]])
a[[1,2],[2,0]] == array([5, 4])


numpy operations
###############################################################################

a.view()			#in-place shallow copy 	 return a view of a reference to same object
a[]				#in-place                return new view
a.copy()			#make a deep copy        return a copy



a.reshape(newshape)     	#in-place    		 return new view
a.ravel()              		#in-place    		 return new view
a.flatten()            	        #make a copy 		 return a copy
a.flat                  	#a.flat == iter(a.flat)  return a iterator
a.T / np.transpose(object)      #in-place                return new view
np.swapaxes(object,axis1,axis2) #in-place                return new view
np.broadcast(x,y)               #(x[0],y[0]),(x[0],y[1]) return iterator
np.expand_dims(object,axis)     #in-place                return new view
np.squeeze(object,axis)         #axis: tuple indicates subset of dim1 axes
np.concatenate((a1,a2..),axis)
np.stack((a1,a2),axis)          #axis 表示新增的那一維, 從 last dims 開始讀取
np.h/vstack((a1,a2))
np.split(arr,  indices_or_sections, axis)
np.resize(arr,shape)            #make a copy             return a copy
np.append(arr,values,axis)      #如果沒有提供 axis返回展開的一維數組
np.insert(arr,values,axis)      #如果沒有提供 axis返回展開的一維數組
np.unique(arr, return_inverse, return_counts)

#a = np.array([5,2,6,2,7,5,6,8,2,9])
#u,indices,indices2 = np.unique(a,return_inverse = True,return_counts = True)
#u == [2 5 6 7 8 9]
#indices == [1 0 2 0 3 1 2 4 0 5],   u[indices] = a
#indices2 == [3 2 2 1 1 1]           重複的個數



np.sort(a, axis, kind, order)          #kind:'quicksort'<'mergesort'<'heapsort',order:結構化數組'name'
np.argsort()                           #return indexs
np.lexsort()                           #按從左到右優先級排序
np.nonzero()                           #a=[[30 40  0],[0,20,10]],np.nonzero(a)==(array([0, 0, 1, 1]), array([0, 1, 1, 2]))
np.where(x > 3)                        #return y 為類似上一層的特殊 Index tuple, x[y]=滿足條件所有元素
np.extract(condition, x)               #condition = np.mod(x,2)==0, condition == [[T,F,T],[F,T,F]],

$$$x[np.where(condition)] == x[condition]




numpy math
###############################################################################
np.sin/cos/tan
np.add/subtract/multipy/divide
np.reciprocal
np.power
np.mod
np.amin/amax/ptp(arr,axis)             #ptp 極值差
np.argmin/max
np.percentile(arr, q, axis)            #q 百分數
np.median/mean/std/var
np.average(arr,weights)                #加權平均
matlib.empty(shape,dtype)/zeros/ones/eye/identity/rand/
np.dot(a,b)
np.matmul(a,b)
np.linalg.det()/.solve(A,B)/.inv()




 

