function P3d = pixel2pts3d(K, u, v, depth)
if(size(u,2) ~= size(v,2) || size(u,2) ~= size(depth,2))
  disp('u, v, depth must be of size (1,n)');
  P3d = zeros(3,0);
else
  oo = ones(1,size(u,2) );
  tt = ones(3,1); 
  
  
  vec = [([u;v] - K(1:2,3) * oo) ./ ([K(1,1);K(2,2)] * oo);oo];

  %normalize the vector
  vec = vec ./ (tt * sqrt(sum(vec.^2) ) );

  %multiply by the depth
  P3d = vec .* (tt * depth);

end