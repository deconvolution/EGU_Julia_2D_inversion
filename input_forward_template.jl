## import packages
using MAT,Plots,Dates,TimerOutputs,WriteVTK,DataFrames,CSV,ProgressMeter

const USE_GPU=false;  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end
include("./seismic2D_function.jl");
Threads.nthreads()
## timing
ti=TimerOutput();
## read stiffness and density
nx=200;
nz=200;
mutable struct C2
    lambda
    mu
    rho
end
vp=@ones(nx,nz)*2000;
vp[:,150:end] .=1500;
vp[:,1:50] .=1200;
vs=@ones(nx,nz)*2000/sqrt(2);
vs[:,150:end] .=1500/sqrt(2);
vs[:,1:50] .=0;
rho=300 *vp .^.25;
rho[:,1:50] .=1000;
C=C2((rho .*vp.^2-2 .*rho .*vs.^2),
(rho .*vs.^2),
(rho));
## define model parameters
dt=.002;
dx=10;
dz=10;
nt=800;

X=(1:nx)*dx;
Z=(1:nz)*dz;

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
s_s1[:] =[50,150];
s_s3[:] .=53;

# source locations true
s_s1t=minimum(X) .+ dx .*s_s1;
s_s3t=dz .*s_s3;

# magnitude
M=2.7;
# source frequency [Hz]
freq=2;

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
## plot
# point interval in time steps, 0 = no plot
plot_interval=100;
# save wavefield
wavefield_interval=0;
## create folder for saving
p2= @__FILE__;
p3=chop(p2,head=0,tail=3);
if isdir(p3)==0
    mkdir(p3);
end
## mute some receiver components
Rm=ones(nt,length(r3),3);
## initialize seismograms
R1=zeros(Float32,nt,length(r3));
R3=copy(R1);
P=copy(R1);
data=zeros(nt,length(r3));
## implement solver
implement_2D_forward(dt,dx,dz,nt,nx,nz,
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
    p3);
## plot seismograms
file=matopen(string(path_rec,"/rec_1.mat"));
tt=read(file,"data");
close(file);
ir=1;
plot(dt:dt:dt*nt,tt[:,ir])
