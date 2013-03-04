mpirun -np 8 python learn_svm_batch.py  --parallel --crop 2 --fold 1 --loss none --loss_factor 1000 --latent --one_iter --duald_niter 10 --duald_gamma 10.0 --Cprime 1.0 -C 0.1 --folder 2013.01.21.exp_latent_DDACI_crop2_Lnone_x1000_Cp1.0_C0.1

mpirun -np 8 python learn_svm_batch.py  --parallel --crop 2 --fold 1 --loss none --loss_factor 1000 --Cprime 1.0 -C 0.1 --folder 2013.01.21.exp_baseline_crop2_Lnone_x1000_Cp1.0_C0.1

mpirun -np 8 python learn_svm_batch.py  --parallel --crop 2 --fold 1 --loss none --loss_factor 1000 --latent --one_iter --duald_niter 10 --duald_gamma 10.0 --Cprime 100.0 -C 0.1 --folder 2013.01.21.exp_latent_DDACI_crop2_Lnone_x1000_Cp100.0_C0.1

mpirun -np 8 python learn_svm_batch.py  --parallel --crop 2 --fold 1 --loss none --loss_factor 1000 --Cprime 100.0 -C 0.1 --folder 2013.01.21.exp_baseline_crop2_Lnone_x1000_Cp100.0_C0.1

mpirun -np 8 python learn_svm_batch.py  --parallel --crop 2 --fold 1 --loss none --loss_factor 1000 --latent --one_iter --duald_niter 10 --duald_gamma 10.0 --Cprime 1000000.0 -C 0.1 --folder 2013.01.21.exp_latent_DDACI_crop2_Lnone_x1000_Cp1000000.0_C0.1

mpirun -np 8 python learn_svm_batch.py  --parallel --crop 2 --fold 1 --loss none --loss_factor 1000 --Cprime 1000000.0 -C 0.1 --folder 2013.01.21.exp_baseline_crop2_Lnone_x1000_Cp1000000.0_C0.1

