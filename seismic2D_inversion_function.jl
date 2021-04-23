@timeit ti "iso_2D" function iso_adjoint_2D(dt,dx,dz,nt,
nx,nz,X,Z,s1,s3,src1,src3,srcp,
s1t,s3t,
lp,nPML,Rc,
C,
plot_interval,
wavefield_interval,
path,
path_pic_adjoint,
path_wavefield_adjoint)

d0=Dates.now();
# source number
ns=length(s3);

#create folder for figures
if isdir(path)==0
    mkdir(path);
end

n_picture=1;
n_wavefield=1;
if plot_interval!=0 && path_pic_adjoint!=nothing
    if isdir(path_pic_adjoint)==0;
        mkdir(path_pic_adjoint);
    end
    # initialize pvd
    pvd=paraview_collection(string(path,"/time_info_adjoint"));
end

if wavefield_interval!=0 && path_wavefield_adjoint!=nothing
    if isdir(path_wavefield_adjoint)==0
        mkdir(path_wavefield_adjoint);
    end
end

# PML
vmax=sqrt.((C.lambda+2*C.mu) ./C.rho);
beta0=(ones(nx,nz) .*vmax .*(nPML+1) .*log(1/Rc)/2/lp/dx);
beta1=(@zeros(nx,nz));
beta3=copy(beta1);
tt=(1:lp)/lp;
tt2=repeat(reshape(tt,lp,1),1,nz);
plane_grad1=@zeros(nx,nz);
plane_grad3=copy(plane_grad1);

plane_grad1[2:lp+1,:]=reverse(tt2,dims=1);
plane_grad1[nx-lp:end-1,:]=tt2;
plane_grad1[1,:]=plane_grad1[2,:];
plane_grad1[end,:]=plane_grad1[end-1,:];

tt2=repeat(reshape(tt,1,lp),nx,1);
plane_grad3[:,2:lp+1]=reverse(tt2,dims=2);
plane_grad3[:,nz-lp:end-1]=tt2;
plane_grad3[:,1]=plane_grad3[:,2];
plane_grad3[:,end]=plane_grad3[:,end-1];

beta1=beta0.*plane_grad1.^nPML;
beta3=beta0.*plane_grad3.^nPML;

IND=unique(findall(f-> f!=0,beta1.*beta3));
beta=beta1+beta3;
beta[IND]=beta[IND]/2;

beta1=beta3=plane_grad1=plane_grad3=vmax=nothing;

# receiver configuration
R1=@zeros(nt,length(r3));
R3=@zeros(nt,length(r3));
P=@zeros(nt,length(r3));

# wave vector
v1=@zeros(nx,nz);
v3=@zeros(nx,nz);

sigmas11=@zeros(nx,nz);
sigmas13=@zeros(nx,nz);
sigmas33=@zeros(nx,nz);
p=@zeros(nx,nz);

l=1;
# save wavefield
if path_wavefield_adjoint!=nothing && wavefield_interval!=0
    global data;
    data=zeros(nx,nz);
    write2mat(string(path_wavefield_adjoint,"/v1_",l,".mat"),data);
    write2mat(string(path_wavefield_adjoint,"/v3_",l,".mat"),data);
    write2mat(string(path_wavefield_adjoint,"/sigmas11_",l,".mat"),data);
    write2mat(string(path_wavefield_adjoint,"/sigmas33_",l,".mat"),data);
    write2mat(string(path_wavefield_adjoint,"/sigmas13_",l,".mat"),data);
    write2mat(string(path_wavefield_adjoint,"/p_",l,".mat"),data);
end
#
v1_3_2_end=@zeros(nx,nz-1);
v3_1_2_end=@zeros(nx-1,nz);
auxiliary_in_vadjoint_1_2_end=@zeros(nx-1,nz);
auxiliary2_in_vadjoint_3_2_end=@zeros(nx,nz-1);
auxiliary_in_vadjoint=@zeros(nx,nz);
auxiliary2_in_vadjoint=@zeros(nx,nz);

pro_bar=Progress(nt,1,"adjoint_simulation...",50);
dlambda=@zeros(nx,nz);
dmu=copy(dlambda);
drho=copy(dlambda);
for l=1:nt-1
    global data
    @timeit ti "shift coordinate" @parallel (2:nx-1) z_2_end(v1,v1_3_2_end);
    @timeit ti "shift coordinate" @parallel (2:nz-1) x_2_end(v3,v3_1_2_end);

    @timeit ti "compute_sigma" @parallel compute_sigma_iso_adjoint(dt,dx,dz,v1,v1_3_2_end,
    v3,v3_1_2_end,
    sigmas11,sigmas13,sigmas33,p,
    beta);

    @timeit ti "minus" @parallel compute_auxiliary_vadjoint(C.lambda,C.mu,sigmas11,sigmas33,p,auxiliary_in_vadjoint,
    auxiliary2_in_vadjoint);

    @timeit ti "shift coordinate" @parallel (2:nx-1) z_2_end(auxiliary2_in_vadjoint,auxiliary2_in_vadjoint_3_2_end);
    @timeit ti "shift coordinate" @parallel (2:nz-1) x_2_end(auxiliary_in_vadjoint,auxiliary_in_vadjoint_1_2_end);

    @timeit ti "compute_v" @parallel compute_v_iso_adjoint(dt,dx,dz,C.rho,C.mu,v1,v3,beta,sigmas13,
    auxiliary_in_vadjoint_1_2_end,auxiliary2_in_vadjoint_3_2_end);


    if ns==1
        v1[CartesianIndex.(s1,s3)]=v1[CartesianIndex.(s1,s3)]+1 ./C.rho[CartesianIndex.(s1,s3)] .*src1[l];
        v3[CartesianIndex.(s1,s3)]=v3[CartesianIndex.(s1,s3)]+1 ./C.rho[CartesianIndex.(s1,s3)] .*src3[l];
        p[CartesianIndex.(s1,s3)]=p[CartesianIndex.(s1,s3)]+srcp[l];
    else
        v1[CartesianIndex.(s1,s3)]=v1[CartesianIndex.(s1,s3)]+1 ./C.rho[CartesianIndex.(s1,s3)] .*src1[l,:]';
        v3[CartesianIndex.(s1,s3)]=v3[CartesianIndex.(s1,s3)]+1 ./C.rho[CartesianIndex.(s1,s3)] .*src3[l,:]';
        p[CartesianIndex.(s1,s3)]=p[CartesianIndex.(s1,s3)]+srcp[l,:]';
    end

    # save wavefield
    if path_wavefield_adjoint!=nothing && wavefield_interval!=0
        data=v1;
        write2mat(string(path_wavefield_adjoint,"/v1_",l+1,".mat"),data);
        data=v3;
        write2mat(string(path_wavefield_adjoint,"/v3_",l+1,".mat"),data);
        data=sigmas11;
        write2mat(string(path_wavefield_adjoint,"/sigmas11_",l+1,".mat"),data);
        data=sigmas33;
        write2mat(string(path_wavefield_adjoint,"/sigmas33_",l+1,".mat"),data);
        data=sigmas13;
        write2mat(string(path_wavefield_adjoint,"/sigmas13_",l+1,".mat"),data);
        data=p;
        write2mat(string(path_wavefield_adjoint,"/p_",l+1,".mat"),data);
    end

    # plot
    if plot_interval!=0 && path_pic_adjoint!=nothing
        if mod(l,plot_interval)==0 || l==nt-1
            vtkfile = vtk_grid(string(path_pic_adjoint,"/wavefield_pic_",n_picture),X,Z);
            vtkfile["v1"]=v1;
            vtkfile["v3"]=v3;
            vtkfile["p"]=p;
            pvd[dt*(l+1)]=vtkfile;
            n_picture=n_picture+1;
        end
    end

    # compute compute_gradient
    if l>=2
        # forward
        data=readmat(string(path,"/forward_wavefield/v1_",nt-l+1,".mat"),"data");
        v1f=data;
        data=readmat(string(path,"/forward_wavefield/v3_",nt-l+1,".mat"),"data");
        v3f=data;
        data=readmat(string(path,"/forward_wavefield/p_",nt-l+1,".mat"),"data");
        pf=data;
        data=readmat(string(path,"/forward_wavefield/v1_",nt-l+2,".mat"),"data");
        v1f2=data;
        data=readmat(string(path,"/forward_wavefield/v3_",nt-l+2,".mat"),"data");
        v3f2=data;
        @parallel compute_gradient_2D(dt,dx,dz,dlambda,dmu,drho,p,v1f,v3f,v1f2,v3f2,
        sigmas11,sigmas13,sigmas33,v1,v3);
    end
    next!(pro_bar);
end
if plot_interval!=0 && path_pic_adjoint!=nothing
    vtk_save(pvd);
end
return dlambda,dmu,drho
end

## compute sigma adjoint
@parallel function compute_sigma_iso_adjoint(dt,dx,dz,v1,v1_3_2_end,
    v3,v3_1_2_end,
    sigmas11,sigmas13,sigmas33,p,
    beta)

    @inn(sigmas11)=dt*@d_xi(v1)/dx+
    @inn(sigmas11)-
    dt*@all(beta) .*@inn(sigmas11);

    @inn(sigmas33)=dt*@d_yi(v3)/dz+
    @inn(sigmas33)-
    dt*@all(beta).*@inn(sigmas33);

    @inn(sigmas13)=dt*(@d_yi(v1_3_2_end)/dz+
    @d_xi(v3_1_2_end)/dx)+
    @inn(sigmas13)-
    dt*@all(beta).*@inn(sigmas13);

    # p
    @inn(p)=-dt*(@d_xi(v1)/dx+@d_yi(v3)/dz)+
    @inn(p)-
    dt*@all(beta).*@inn(p);
    return nothing
end

## compute v adjoint
@parallel function compute_v_iso_adjoint(dt,dx,dz,rho,mu,v1,v3,beta,sigmas13,
    auxiliary_in_vadjoint_1_2_end,auxiliary2_in_vadjoint_3_2_end)

    @inn(v1)=dt./@all(rho) .*(@all(mu) .*@d_yi(sigmas13)/dz+
    @d_xi(auxiliary_in_vadjoint_1_2_end)/dx)+
    @inn(v1)-
    dt*@all(beta) .*@inn(v1);

    @inn(v3)=dt./@all(rho) .*(@all(mu) .*@d_xi(sigmas13)/dx+
    @d_yi(auxiliary2_in_vadjoint_3_2_end)/dz)+
    @inn(v3)-
    dt*@all(beta) .*@inn(v3);
    return nothing
end
##
@parallel function compute_auxiliary_vadjoint(lambda,mu,sigmas11,sigmas33,p,auxiliary_in_vadjoint,
    auxiliary2_in_vadjoint)
    @inn(auxiliary_in_vadjoint)=@all(mu) .*@inn(sigmas11)-
    @all(mu) .*@inn(sigmas33)-(@all(lambda)+@all(mu)) .*@inn(p);
    @inn(auxiliary2_in_vadjoint)=@all(mu).*@inn(sigmas33)-
    @all(mu) .*@inn(sigmas11)-(@all(lambda)+@all(mu)) .*@inn(p);
    return nothing
end
##
@parallel function compute_gradient_2D(dt,dx,dz,dlambda,dmu,drho,pa,v1f,v3f,v1f2,v3f2,
    sigmas11a,sigmas13a,sigmas33a,v1a,v3a)
    @inn(dlambda)=@inn(dlambda)+@inn(pa) .*(@d_xi(v1f)/dx+@d_yi(v3f)/dz);

    @inn(dmu)=@inn(dmu)+@d_xi(v1f)/dx .*(@inn(pa)-@inn(sigmas11a)+@inn(sigmas33a))+
    @d_yi(v1f)/dz.* -@inn(sigmas13a)+
    @d_xi(v3f)/dx .* -@inn(sigmas13a)+
    @d_yi(v3f)/dz .* (@inn(pa)+@inn(sigmas11a)-@inn(sigmas33a));

    @inn(drho)=@inn(drho)+@inn(v1a).*(@inn(v1f2)-@inn(v1f))/dt+
    @inn(v3a).*(@inn(v3f2)-@inn(v3f))/dt;
    return nothing
end
##
function taperr(nx,nz,dx,dz,r1,r3,a,b)
    tpr=ones(nx,nz);
    i1,i3=meshgrid(1:nx,1:nz);
    for ir=1:length(r3)
        r=sqrt.(((i1 .-r1[ir]) .*dx) .^2+((i3 .-r3[ir]) .*dz) .^2);
        tpr=tpr .*1 ./sqrt.(1 .+a .*exp.(-b .*r) .^2);
    end
    return tpr
end

function tapers(nx,nz,dx,dz,s1,s3,a,b)
    tps=ones(nx,nz);
    i1,i3=meshgrid(1:nx,1:nz);
    for is=1:length(s3)
        r=sqrt.(((i1 .-s1[is]) .*dx) .^2+((i3 .-s3[is]) .*dz) .^2);
        tps=tps .*1 ./sqrt.(1 .+a .*exp.(-b .*r));
    end
    return tps
end
