template <typename DType, int NDim>
int points_to_voxel_3d_np(py::array_t<DType> points,            // [N,5]
                          py::array_t<DType> voxels,            // [60000,10,5]
                          py::array_t<DType> voxel_point_mask,  // [60000,10]
                          py::array_t<int> coors,               // [60000,3]
                          py::array_t<int> num_points_per_voxel,// [60000]
                          py::array_t<int> coor_to_voxelidx,    // 维度[40,1440,1440] -1填充
                          std::vector<DType> voxel_size,        // [0.075, 0.075, 0.2]
                          std::vector<DType> coors_range,       // [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0] 
                          int max_points,                       // 10
                          int max_voxels                        // 60000
                          ) {
  auto points_rw = points.template mutable_unchecked<2>(); // Will throw if ndim != 2 or flags.writeable is false
  auto voxels_rw = voxels.template mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false
  auto voxel_point_mask_rw = voxel_point_mask.template mutable_unchecked<2>();
  auto coors_rw = coors.mutable_unchecked<2>();     // Will throw if ndim != 2 or flags.writeable is false
  auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>(); // Will throw if ndim != 1 or flags.writeable is false
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>(); // Will throw if ndim != 3 or flags.writeable is false
  auto N = points_rw.shape(0);                      // N
  auto num_features = points_rw.shape(1);           // 4
  // auto ndim = points_rw.shape(1) - 1;
  constexpr int ndim_minus_1 = NDim - 1;            // 2
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim];                                   // int coor[3]
  int c;
  int grid_size[NDim];                              // int grid_size[3]
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] = round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]); // [1440,1440,40]
  }
  int voxelidx, num;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);  // voxel网格坐标
      if ((c < 0 || c >= grid_size[j])) {           // 超出坐标范围
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c; // z,y,x
    }
    if (failed) // 该点超出范围
      continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;                           // voxel索引，代表第几个voxel
      if (voxel_num >= max_voxels)
        continue;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        coors_rw(voxelidx, k) = coor[k];              // z,y,x voxel网格坐标
      }
    }
    num = num_points_per_voxel_rw(voxelidx);          // voxel的点个数，初始为0
    if (num < max_points) {                           // 10
      voxel_point_mask_rw(voxelidx, num) = DType(1);  // 1
      for (int k = 0; k < num_features; ++k) {        // 特征维度遍历
        voxels_rw(voxelidx, num, k) = points_rw(i, k);// voxel特征[60000,10,5]
      }
      num_points_per_voxel_rw(voxelidx) += 1;
    }
  }
  for (int i = 0; i < voxel_num; ++i) {
    coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1; // 对存在的voxel网格坐标 coor_to_voxelidx_rw 取1
  }
  return voxel_num;
}
