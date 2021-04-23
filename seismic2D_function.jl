

function implement_2D_forward(dt,dx,dz,nt,nx,nz,
    X,Z,
    r1,r3,
    Rm,
    s_s1,s_s3,s_src1,s_src3,s_source_type,
    r1t,r3t,
    s_s1t,s_s3t,
    lp,nPML,Rc,
    C,
    plot_interval,
    wavefield_interval,
    p3)

    global path,v1,v3,R1,R3,P

    # initialize seismograms
    R1=zeros(Float32,nt,length(r3));
    R3=copy(R1);
    P=copy(R1);
    data=zeros(nt,length(r3));

    ##
    @time begin
        if msot==1
            global s1,s3,s1t,s3t,src1,src3,source_type,v1,v3,R1,R3,P,path,data,
            path_pic,path_model,path_wavefield,path_rec
            # path for this source
            path=p3;
            if isdir(string(path))==0
                mkdir(string(path));
            end;
            path_pic=string(path,"/pic");
            path_model=string(path,"/model");
            path_wavefield=string(path,"/wavefield");
            path_rec=string(path,"/rec");

            # source locations
            s1=s_s1;
            s3=s_s3;

            s1t=s_s1t;
            s3t=s_s3t;

            src1=s_src1;
            src3=s_src3;
            source_type=s_source_type;

            # pass parameters to solver
            v1,v3,R1,R3,P=iso_2D(dt,dx,dz,nt,
            nx,nz,X,Z,r1,r3,s1,s3,src1,src3,source_type,
            r1t,r3t,
            Rm,
            s1t,s3t,
            lp,nPML,Rc,
            C,
            plot_interval,
            wavefield_interval,
            path,
            path_pic,
            path_model,
            path_wavefield,
            path_rec);


        else
            for source_code=1:length(s_s3)
                global s1,s3,s1t,s3t,src1,src3,source_type,v1,v3,R1,R3,P,path,data,
                path_pic,path_model,path_wavefield,path_rec
                # source locations
                s1=s_s1[source_code];
                s3=s_s3[source_code];

                # path for this source
                path=string(p3,"/source_code_",
                (source_code));

                path_pic=string(path,"/pic");
                path_model=string(path,"/model");
                path_wavefield=string(path,"/wavefield");
                path_rec=string(path,"/rec");

                s1=s_s1[source_code];
                s3=s_s3[source_code];

                s1t=s_s1t[source_code];
                s3t=s_s3t[source_code];

                src1=s_src1[:,source_code];
                src3=s_src3[:,source_code];
                source_type=string(s_source_type[source_code]);


                # pass parameters to solver
                v1,v3,R1,R3,P=iso_2D(dt,dx,dz,nt,
                nx,nz,X,Z,r1,r3,s1,s3,src1,src3,source_type,
                r1t,r3t,
                Rm,
                s1t,s3t,
                lp,nPML,Rc,
                C,
                plot_interval,
                wavefield_interval,
                path,
                path_pic,
                path_model,
                path_wavefield,
                path_rec);
            end
        end
    end
end

function meshgrid(x,y)
    x2=zeros(length(x),length(y));
    y2=x2;
    x2=repeat(x,1,length(y));
    y2=repeat(reshape(y,1,length(y)),length(x),1);
    return x2,y2
end

function write2mat(path,var)
    file=matopen(path,"w");
    write(file,"data",data);
    close(file);
    return nothing
end

function readmat(path,var)
    file=matopen(path);
    tt=read(file,var);
    close(file)
    return tt
end

function rickerWave(freq,dt,ns,M)
    ## calculate scale
    E=10 .^(5.24+1.44 .*M);
    s=sqrt(E.*freq/.299);

    t=dt:dt:dt*ns;
    t0=1 ./freq;
    t=t .-t0;
    ricker=s .*(1 .-2*pi^2*freq .^2*t .^2).*exp.(-pi^2*freq^2 .*t .^2);
    ricker=ricker;
    ricker=Float32.(ricker);
    return ricker
end
##
@parallel function compute_sigma(dt,dx,dz,lambda,mu,beta,v1,v1_3_2_end,
    v3,v3_1_2_end,
    sigmas11,sigmas13,sigmas33,p)

    @inn(sigmas11)=dt*.5*(2*@all(mu) .*@d_xi(v1)/dx+
    (-2*@all(mu)) .*@d_yi(v3)/dz)+
    @inn(sigmas11)-
    dt*@all(beta) .*@inn(sigmas11);

    @inn(sigmas33)=dt*.5*(2*@all(mu) .*@d_yi(v3)/dz+
    (-2*@all(mu)) .*@d_xi(v1)/dx)+
    @inn(sigmas33)-
    dt*@all(beta).*@inn(sigmas33);

    @inn(sigmas13)=dt*(@all(mu) .*(@d_yi(v1_3_2_end)/dz+
    @d_xi(v3_1_2_end)/dx))+
    @inn(sigmas13)-
    dt*@all(beta).*@inn(sigmas13);

    # p
    @inn(p)=-dt*((@all(lambda)+@all(mu)) .*@d_xi(v1)/dx+
    (@all(lambda)+@all(mu)) .*@d_yi(v3)/dz)+
    @inn(p)-
    dt*@all(beta).*@inn(p);

    return nothing
end

@parallel_indices (iz) function x_2_end(in,out)
out[:,iz]=in[2:end,iz];
return nothing
end

@parallel_indices (ix) function z_2_end(in,out)
out[ix,:]=in[ix,2:end];
return nothing
end
##
@parallel function compute_v(dt,dx,dz,rho,v1,v3,beta,sigmas11_minus_p_1_2_end,
    sigmas13,sigmas33_minus_p_3_2_end)

    @inn(v1)=dt./@all(rho) .*(@d_xi(sigmas11_minus_p_1_2_end)/dx+
    @d_yi(sigmas13)/dz)+
    @inn(v1)-
    dt*@all(beta) .*@inn(v1);

    @inn(v3)=dt./@all(rho) .*(@d_xi(sigmas13)/dx+
    @d_yi(sigmas33_minus_p_3_2_end)/dz)+
    @inn(v3)-
    dt*@all(beta) .*@inn(v3);
    return nothing
end
@parallel function minus(a,b,c)
    @all(c)=@all(a)-@all(b);
    return nothing
end

@timeit ti "iso_2D" function iso_2D(dt,dx,dz,nt,
nx,nz,X,Z,r1,r3,s1,s3,src1,src3,source_type,
r1t,r3t,
Rm,
s1t,s3t,
lp,nPML,Rc,
C,
plot_interval,
wavefield_interval,
path,
path_pic,
path_model,
path_wavefield,
path_rec)

global data

d0=Dates.now();
# source number
ns=length(s3);

# create main folder
if isdir(path)==0
    mkdir(path);
end

# create folder for picture
n_picture=1;
n_wavefield=1;
if path_pic!=nothing
    if isdir(path_pic)==0
        mkdir(path_pic);
    end
    # initialize pvd
    pvd=paraview_collection(string(path,"/time_info_forward"));
end

# create folder for model
if path_model!=nothing
    if isdir(path_model)==0
        mkdir(path_model)
    end
    vtkfile = vtk_grid(string(path_model,"/model"),X,Z);
    vtkfile["lambda"]=C.lambda;
    vtkfile["mu"]=C.mu;
    vtkfile["rho"]=C.rho;

    data=C.lambda;
    write2mat(string(path_model,"/lambda.mat"),data);
    data=C.mu;
    write2mat(string(path_model,"/mu.mat"),data);
    data=C.rho;
    write2mat(string(path_model,"/rho.mat"),data);

    vtk_save(vtkfile);
    CSV.write(string(path_model,"/receiver location.csv"),DataFrame([r1t' r3t']));
    CSV.write(string(path_model,"/source location.csv"),DataFrame([s1t' s3t']));
end

# create folder for wavefield
if path_wavefield!=nothing
    if isdir(path_wavefield)==0
        mkdir(path_wavefield);
    else
        rm(path_wavefield,recursive=true);
        mkdir(path_wavefield);
    end
end


# create folder for rec
if path_rec!=nothing
    if isdir(path_rec)==0
        mkdir(path_rec)
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
if path_wavefield!=nothing && wavefield_interval!=0
    if mod(l,wavefield_interval)==0
        data=zeros(nx,nz);
        write2mat(string(path_wavefield,"/v1_",n_wavefield,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path_wavefield,"/v3_",n_wavefield,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path_wavefield,"/sigmas11_",n_wavefield,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path_wavefield,"/sigmas33_",n_wavefield,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path_wavefield,"/sigmas13_",n_wavefield,".mat"),data);
        data=zeros(nx,nz);
        write2mat(string(path_wavefield,"/p_",n_wavefield,".mat"),data);
        n_wavefield=n_wavefield+1;
    end
end
#
v1_3_2_end=@zeros(nx,nz-1);
v3_1_2_end=@zeros(nx-1,nz);
sigmas11_minus_p_1_2_end=@zeros(nx-1,nz);
sigmas33_minus_p_3_2_end=@zeros(nx,nz-1);
sigmas11_minus_p=@zeros(nx,nz);
sigmas33_minus_p=@zeros(nx,nz);
pro_bar=Progress(nt,1,"forward_simulation...",50);
for l=1:nt-1
    @timeit ti "shift coordinate" @parallel (2:nx-1) z_2_end(v1,v1_3_2_end);
    @timeit ti "shift coordinate" @parallel (2:nz-1) x_2_end(v3,v3_1_2_end);
    @timeit ti "compute_sigma" @parallel compute_sigma(dt,dx,dz,C.lambda,C.mu,beta,v1,v1_3_2_end,
    v3,v3_1_2_end,
    sigmas11,sigmas13,sigmas33,p);

    @timeit ti "minus" @parallel minus(sigmas11,p,sigmas11_minus_p);
    @timeit ti "minus" @parallel minus(sigmas33,p,sigmas33_minus_p);

    @timeit ti "shift coordinate" @parallel (2:nz-1) x_2_end(sigmas11_minus_p,sigmas11_minus_p_1_2_end);
    @timeit ti "shift coordinate" @parallel (2:nx-1) z_2_end(sigmas33_minus_p,sigmas33_minus_p_3_2_end);

    @timeit ti "compute_v" @parallel compute_v(dt,dx,dz,C.rho,v1,v3,beta,sigmas11_minus_p_1_2_end,
    sigmas13,sigmas33_minus_p_3_2_end);

    @timeit ti "source" if source_type=="D"
    if ns==1
        v1[CartesianIndex.(s1,s3)]=v1[CartesianIndex.(s1,s3)]+1 ./C.rho[CartesianIndex.(s1,s3)] .*src1[l];
        v3[CartesianIndex.(s1,s3)]=v3[CartesianIndex.(s1,s3)]+1 ./C.rho[CartesianIndex.(s1,s3)] .*src3[l];
    else
        v1[CartesianIndex.(s1,s3)]=v1[CartesianIndex.(s1,s3)]+1 ./C.rho[CartesianIndex.(s1,s3)] .*src1[l,:]';
        v3[CartesianIndex.(s1,s3)]=v3[CartesianIndex.(s1,s3)]+1 ./C.rho[CartesianIndex.(s1,s3)] .*src3[l,:]';
    end
end

@timeit ti "source" if source_type=="P"
if ns==1
    p[CartesianIndex.(s1,s3)]=p[CartesianIndex.(s1,s3)]+src3[l];
else
    p[CartesianIndex.(s1,s3)]=p[CartesianIndex.(s1,s3)]+src3[l,:]';
end
end

# assign recordings
@timeit ti "receiver" R1[l+1,:]=reshape(v1[CartesianIndex.(r1,r3)],length(r3),);
@timeit ti "receiver" R3[l+1,:]=reshape(v3[CartesianIndex.(r1,r3)],length(r3),);
@timeit ti "receiver" P[l+1,:]=reshape(p[CartesianIndex.(r1,r3)],length(r3),);
# save wavefield
if path_wavefield!=nothing && wavefield_interval!=0
    if mod(l,wavefield_interval)==0
        data=v1;
        write2mat(string(path_wavefield,"/v1_",n_wavefield,".mat"),data);
        data=v3;
        write2mat(string(path_wavefield,"/v3_",n_wavefield,".mat"),data);
        data=sigmas11;
        write2mat(string(path_wavefield,"/sigmas11_",n_wavefield,".mat"),data);
        data=sigmas33;
        write2mat(string(path_wavefield,"/sigmas33_",n_wavefield,".mat"),data);
        data=sigmas13;
        write2mat(string(path_wavefield,"/sigmas13_",n_wavefield,".mat"),data);
        data=p;
        write2mat(string(path_wavefield,"/p_",n_wavefield,".mat"),data);
        n_wavefield=n_wavefield+1;
    end
end

# plot
if path_pic!=nothing && plot_interval!=0
    if mod(l,plot_interval)==0 || l==nt-1
        vtkfile = vtk_grid(string(path_pic,"/wavefield_pic_",n_picture),X,Z);
        vtkfile["v1"]=v1;
        vtkfile["v3"]=v3;
        vtkfile["p"]=p;
        pvd[dt*(l+1)]=vtkfile;
        n_picture=n_picture+1;
    end
end
next!(pro_bar);
end

R1=R1 .*Rm[:,:,1];
R3=R3 .*Rm[:,:,2];
P=P .*Rm[:,:,3];

data=R1;
write2mat(string(path_rec,"/rec_1.mat"),data);
data=R3;
write2mat(string(path_rec,"/rec_3.mat"),data);
data=P;
write2mat(string(path_rec,"/rec_p.mat"),data);

if path_pic!=nothing && plot_interval!=0
vtk_save(pvd);
end

return v1,v3,R1,R3,P
end
