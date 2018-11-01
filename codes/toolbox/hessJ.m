function HJ = hessJ(h,U,V,S,mu)

[n,n] = size(U);
HJ = zeros(n,n);
hb = U'*h*V;

for i=1:n
  for j=1:n
    if (S(i) | S(j)) & (i ~= j)
      HJ(i,j) = (hb(i,j)-hb(j,i))/(S(j)+S(i));
      %HJ(i,j) = (S(j)*(hb(i,j)-hb(j,i))+S(i)*(hb(j,i)-hb(i,j)))/(S(j)^2-S(i)^2);
      %[i j HJ(i,j)]
      %pause
    end
  end
end

HJ = mu*U*HJ*V';
