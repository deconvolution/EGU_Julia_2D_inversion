## import packages
using MAT,Plots,Dates,TimerOutputs,WriteVTK,DataFrames,CSV,LinearAlgebra,ProgressMeter

const USE_GPU=false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end
include("./seismic2D_function.jl");
include("./seismic2D_inversion_function.jl");
Threads.nthreads()
## timing
ti=TimerOutput();
## define model parameters
nx=200;
nz=200;
dt=.002;
dx=10;
dz=10;
nt=800;

mutable struct C2
    lambda
    mu
    rho
end
vp=@ones(nx,nz)*2000;
vp[:,1:50] .=1200;
vs=@ones(nx,nz)*2000/sqrt(2);
vs[:,1:50] .=0;
rho=300 *vp .^.25;
rho[:,1:50] .=1000;
C=C2((rho .*vp.^2-2 .*rho .*vs.^2),
(rho .*vs.^2),
(rho));

X=(1:nx)*dx;
Z=(1:nz)*dz;
## assign stiffness matrix and rho
# PML layers
lp=20;

# PML coefficient, usually 2
nPML=2;

# Theoretical coefficient, more PML layers, less R
# Empirical values
# lp=[10,20,30,40]
# R=[.1,.01,.001,.0001]
Rc=.0001;
## source
# source location
# multiple source at one time or one source at each time
msot=0;

# source location grid
s_s1=zeros(Int32,1,2);
s_s3=copy(s_s1);
s_s1[:] =[50 150];
s_s3[:] .=53;

# source locations true
s_s1t=minimum(X) .+ dx .*s_s1;
s_s3t=dz .*s_s3;

# magnitude
M=2.7;
# source frequency [Hz]
freq=5;


# source signal
singles=rickerWave(freq,dt,nt,M);

# give source signal to x direction
s_src1=zeros(Float32,nt,length(s_s3));

# give source signal to z direction
s_src3=copy(s_src1);

for i=1:length(s_s3)
s_src3[:,i]=singles;
end

# source type. 'D' for directional source. 'P' for P-source.
s_source_type="D"^length(s_s3);
## receiver
r1t=zeros(Float32,1,81);
r3t=copy(r1t);
r1=zeros(Int32,1,size(r1t,2));
r3=copy(r1);
r1[:]=20:2:nx-20;
r3[:] .=53;

r1t[:] =r1 .*dx;
r3t[:] =r3 .*dz;
## create folder for saving
p2= @__FILE__;
if isdir(chop(p2,head=0,tail=3))==0
    mkdir(chop(p2,head=0,tail=3));
end
p3=chop(p2,head=0,tail=3);
if isdir(string(p3,"/inversion_process/"))==0
    mkdir(string(p3,"/inversion_process/"));
end
# point interval in time steps, 0 = no plot
plot_interval=100;
# save wavefield
wavefield_interval=1;
path=p3;

path_true="./input_forward_template";

path_pic_forward=string(p3,"/forward_pic");
path_pic_adjoint=string(p3,"/adjoint_pic");
path_wavefield_forward=string(p3,"/forward_wavefield");
path_wavefield_adjoint=nothing;
## inversion
nit=1;
mlambda=maximum(C.lambda)/20;
mmu=maximum(C.mu)/20;
mrho=maximum(C.rho)/100;

global cost
cost=zeros(nit,1);

v1f=zeros(nx,nz);
v3f=copy(v1f);
v1a=copy(v1f);
v3a=copy(v1f);
pf=copy(v1f);
pa=copy(v1f);
n_source_code=length(s_s3);
## mute some receiver components
Rm=ones(nt,length(r3),3);
## mute correction where properties are known
top_layer=50;
cm=meshgrid(1:nx,1:top_layer);
cm=CartesianIndex.(cm[1],cm[2]);
##
for it=1:nit
    println("\niteration=",it,"/",nit);
    global data;
    # create folder to save inversion result
    p4=string(path,"/inversion_process/inversion_iteration_",it);
    if isdir(p4)==0
        mkdir(p4);
    end
    path_model=string(p4,"/model");

    for source_code=1:(n_source_code)
        global s1,s3,s1t,s3t,src1,src3,source_type,v1,v3,R1,R3,P,path,
        dlambda,dmu,drho
        dlambda=@zeros(nx,nz);
        dmu=copy(dlambda);
        drho=copy(dlambda);

        path_rec=string(p4,"/source_code_",source_code,"/rec");

        data=readmat(string(path_true,"/source_code_",source_code,
        "/rec/rec_p.mat"),"data");
        tP=data;
        data=readmat(string(path_true,"/source_code_",source_code,
        "/rec/rec_1.mat"),"data");
        tR1=data;
        data=readmat(string(path_true,"/source_code_",source_code,
        "/rec/rec_3.mat"),"data");
        tR3=data;

        # forward
        # source locations
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
        string(p4,"/source_code_",source_code),
        path_pic_forward,
        path_model,
        path_wavefield_forward,
        path_rec);

        # compare
        misR1=R1-tR1;
        misR3=R3-tR3;
        misP=P-tP;

        # reversal misfit
        rev_misR1=reverse(misR1,dims=1);
        rev_misR3=reverse(misR3,dims=1);
        rev_misP=reverse(misP,dims=1);

        # adjoint simulation
        dlambda,dmu,drho=iso_adjoint_2D(dt,dx,dz,nt,
        nx,nz,X,Z,r1,r3,rev_misR1,rev_misR3,rev_misP,
        r1t,r3t,
        lp,nPML,Rc,
        C,
        plot_interval,
        wavefield_interval,
        path,
        path_pic_adjoint,
        path_wavefield_adjoint);
        dlambda[cm] .=0;
        dmu[cm] .=0;
        drho[cm] .=0;

        a=1200;
        b=.02;
        dlambda=dlambda.*tapers(nx,nz,dx,dz,s1,s3,a,b).*taperr(nx,nz,dx,dz,r1,r3,a,b);

        dmu=dmu.*tapers(nx,nz,dx,dz,s1,s3,a,b).*taperr(nx,nz,dx,dz,r1,r3,a,b);

        drho=drho.*tapers(nx,nz,dx,dz,s1,s3,a,b).*taperr(nx,nz,dx,dz,r1,r3,a,b);
        data=dlambda;
        write2mat(string(path,"/inversion_process/inversion_iteration_",it,
        "/source_code_",source_code,"/dlambda.mat"),data);
        data=dmu;
        write2mat(string(path,"/inversion_process/inversion_iteration_",it,
        "/source_code_",source_code,"/dmu.mat"),data);
        data=drho;
        write2mat(string(path,"/inversion_process/inversion_iteration_",it,
        "/source_code_",source_code,"/drho.mat"),data);
        data=.5*norm(misR1.^2+misR3.^2,2);
        write2mat(string(path,"/inversion_process/inversion_iteration_",it,
        "/source_code_",source_code,"/cost.mat"),data);
    end

    dlambda2=zeros(nx,nz);
    dmu2=copy(dlambda2);
    drho2=copy(dlambda2);
    for is=1:n_source_code
        data=readmat(string(path,"/inversion_process/inversion_iteration_",it,"/source_code_",is,"/dlambda.mat"),"data");
        dlambda2=dlambda2+data;
        data=readmat(string(path,"/inversion_process/inversion_iteration_",it,"/source_code_",is,"/dmu.mat"),"data");
        dmu2=dmu2+data;
        data=readmat(string(path,"/inversion_process/inversion_iteration_",it,"/source_code_",is,"/drho.mat"),"data");
        drho2=drho2+data;
        data=readmat(string(path,"/inversion_process/inversion_iteration_",it,"/source_code_",is,"/cost.mat"),"data");
        cost[it]=cost[it]+data;
    end
    cost[it]=cost[it]/n_source_code;
    # correct
    if it==1
        global alplambda,alpmu,alprho
        alplambda=mlambda ./maximum(abs.(dlambda2));
        if isnan(alplambda) || isinf(alplambda)
            alplambda=0;
        end
        alpmu=mmu ./maximum(abs.(dmu2));
        if isnan(alpmu) || isinf(alpmu)
            alpmu=0;
        end
        alprho=mrho ./maximum(abs.(drho2));
        if isnan(alprho) || isinf(alprho)
            alprho=0;
        end
    end

    C.lambda=C.lambda-alplambda*dlambda2;
    C.mu=C.mu-alplambda*dmu2;
    C.rho=C.rho-alprho*drho2;

    data=C.lambda;
    write2mat(string(path,"/inversion_process/inversion_iteration_",it,"/lambda.mat"),data);

    data=C.mu;
    write2mat(string(path,"/inversion_process/inversion_iteration_",it,"/mu.mat"),data);

    data=C.rho;
    write2mat(string(path,"/inversion_process/inversion_iteration_",it,"/rho.mat"),data);

    vtkfile=vtk_grid(string(path,"/inversion_process/inversion_iteration_",it,"/model"),X,Z);
    vtkfile["lambda"]=C.lambda;
    vtkfile["mu"]=C.mu;
    vtkfile["rho"]=C.rho;
    vtk_save(vtkfile);
end
##
data=C;
write2mat(string(path,"/final_result","C.mat"),data);
data=cost;
write2mat(string(path,"/final_result","final_cost.mat"),data);
