#!/usr/bin/python3

import os
from littoral.system.dap_utils import clustering_models, data_dir, img_dir

if __name__ == '__main__':
  if not (os.path.exists(data_dir) and os.path.isdir(data_dir)):
    os.mkdir(data_dir)
    print(f'Creating dir {data_dir}')
  else:
    print(f'Dir {data_dir} exists')
  
  if not (os.path.exists(img_dir) and os.path.isdir(img_dir)):
    os.mkdir(img_dir)
    print(f'Creating dir {img_dir}')
  else:
    print(f'Dir {img_dir} exists')
  
  confirm = \
    input('Do you want to create subfolders whose names are defined on the variable named clustering_models? (y or n): ')
  
  if(confirm.lower() == 'y'):
    for model in clustering_models:
      data_name = f'{data_dir}/{model}'
      img_name = f'{img_dir}/{model}'
      
      if not (os.path.exists(data_name) and os.path.isdir(data_name)):
        os.mkdir(data_name)
        print(f'Creating dir {data_name}')
      else:
        print(f'Dir {data_name} exists')
      
      if not (os.path.exists(img_name) and os.path.isdir(img_name)):
        os.mkdir(img_name)
        print(f'Creating dir {img_name}')
      else:
        print(f'Dir {img_name} exists')
