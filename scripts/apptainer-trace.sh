apptainer run --nv \
    --bind $(pwd)/experiments:/opt/simulon/experiments \
    --bind $(pwd)/examples:/opt/simulon/examples \
    --bind $(pwd)/vendor:/opt/simulon/vendor \
    --bind $(pwd)/simulon:/opt/simulon/simulon \
    --bind $(pwd)/templates:/opt/simulon/templates \
    --bind $(pwd)/output:/opt/simulon/output \
    simulon-nemo.sif \
    bash -c 'CUDA_DEVICE_MAX_CONNECTIONS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True simulon trace generate "$@"' -- "$@"
