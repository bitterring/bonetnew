# py2.7 + TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py27tf12/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py2.7 + TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py27tf14/lib/python2.7/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /home/bo/miniconda2/envs/py27tf14/lib/python2.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/home/bo/miniconda2/envs/py27tf14/lib/python2.7/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py3.6 + TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/bo/miniconda2/envs/py36tf12/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# py3.6 + TF1.4
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /home/w/anaconda3/envs/py37/lib/python3.7/site-packages/tensorflow/include -I /usr/local/cuda-10.0/include -I /home/w/anaconda3/envs/py37/lib/python3.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-10.0/lib64/ -L/home/w/anaconda3/envs/py37/lib/python3.7/site-packages/tensorflow -ltensorflow_framework -O2 #-D_GLIBCXX_USE_CXX11_ABI=0
