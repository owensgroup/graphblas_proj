
int get_num_gpus() {
  int num_gpus = -1;
  cudaGetDeviceCount(&num_gpus);
  return num_gpus;
}

void enable_peer_access() {
  int num_gpus = get_num_gpus();
  
  for(int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    for(int j = 0; j < num_gpus; j++) {
      if(i == j) 
        continue;
      cudaDeviceEnablePeerAccess(j, 0);
    }
  }
  
  cudaSetDevice(0);
}