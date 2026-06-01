apptainer run --nv \
    --bind $(pwd)/experiments:/opt/simulon/experiments \
    --bind $(pwd)/examples:/opt/simulon/examples \
    --bind $(pwd)/vendor:/opt/simulon/vendor \
    --bind $(pwd)/simulon:/opt/simulon/simulon \
    --bind $(pwd)/templates:/opt/simulon/templates \
    --bind $(pwd)/output:/opt/simulon/output \
    simulon-nemo.sif \
    bash -c "NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 CUDA_DEVICE_MAX_CONNECTIONS=1 simulon trace generate $1 --memory-snapshot output/mem.pickle"

