"""
Off-policy generative adversarial imitation learning (GAIL) solver.

```julia
OffPolicyGAIL(;
    π,
    S, 
    𝒟_demo, 
    𝒟_ndas::Array{ExperienceBuffer} = ExperienceBuffer[], 
    normalize_demo::Bool=true, 
    D::ContinuousNetwork, 
    solver=SAC, 
    d_opt::NamedTuple=(epochs=5,), 
    log::NamedTuple=(;), 
    kwargs...)
```
"""
function OffPolicyGAIL(;
        π,
        S, 
        𝒟_demo, 
        𝒟_ndas::Array{ExperienceBuffer} = ExperienceBuffer[], 
        normalize_demo::Bool=true, 
        add_noise::Bool=false,
        D::ContinuousNetwork, 
        solver=SAC, 
        d_opt::NamedTuple=(epochs=5,), 
        log::NamedTuple=(;), 
        kwargs...)
                        
    # Define the training parameters for the desciminator
    d_opt = TrainingParams(;name="discriminator_", loss=()->nothing, d_opt...)
    
    # Setup NDA parameters
    N_nda = length(𝒟_ndas)
    λ_nda = Float32(-1 / N_nda)
    N_datasets = 2 + N_nda
    
    println("λnda: $(λ_nda), N_datasets:$(N_datasets), N_nda:$(N_nda)")

    # Normalize and/or change device of expert and NDA data
    dev = device(π)
    A = action_space(π)
    normalize_demo && (𝒟_demo = normalize!(deepcopy(𝒟_demo), S, A))
    𝒟_demo = 𝒟_demo |> dev
    for i in 1:N_nda
        𝒟_ndas[i] = normalize_demo && normalize!(deepcopy(𝒟_ndas[i]), S, A)
        𝒟_ndas[i] = 𝒟_ndas[i] |> dev
    end

    # Build the solver
    𝒮 = solver(;π=π, 
                S=S,
                log=(dir="log/offpolicygail", period=500, log...),
                kwargs...)
            
    # Setup the training of the discriminator
    B = d_opt.batch_size
    
    # These are minibatch buffers
    𝒟_batch = buffer_like(𝒮.buffer, capacity=B, device=dev)
    𝒟_demo_batch = deepcopy(𝒟_batch)
    𝒟_demo_π_batch = deepcopy(𝒟_batch)
    𝒟_ndas_batch = [deepcopy(𝒟_batch) for 𝒟_nda in 𝒟_ndas]
    𝒟_ndas_π_batch = [deepcopy(𝒟_batch) for 𝒟_nda in 𝒟_ndas]
    
    function GAIL_callback(𝒟; info=Dict(), kwargs...)
        for i=1:d_opt.epochs
            
            # Sample minibatchs
            rand!(𝒟_demo_batch, 𝒟_demo)
            # rand!(𝒟_demo_π_batch, 𝒟_demo)
            # 𝒟_demo_π_batch.data[:a] = action(π, 𝒟_demo_π_batch[:s])
            rand!(𝒟_batch, 𝒮.buffer)
            for i in 1:N_nda
                rand!(𝒟_ndas_batch[i], 𝒟_ndas[i])
                # rand!(𝒟_ndas_π_batch[i], 𝒟_ndas[i])
                # 𝒟_ndas_π_batch[i].data[:a] = action(π, 𝒟_ndas_π_batch[i][:s])
            end
            # Concatenate minibatches into on buffer
            𝒟_full = hcat(𝒟_demo_batch, 𝒟_batch, 𝒟_ndas_batch...)
            
            # concat inputs
            x = cat(flatten(𝒟_full[:s]), 𝒟_full[:a], dims=1)
            
            # Add some noise (Optional)
            if add_noise
                x .+= Float32.(rand(Normal(0, 0.1f0), size(x))) |> dev
            else
                x = x |> dev
            end
            # Create labels
            y_demo = Flux.onehotbatch(ones(Int, B), 1:N_datasets)
            # y_demo_π = Flux.onehotbatch(2*ones(Int, B), 1:N_datasets)
            y_π = Flux.onehotbatch(2*ones(Int, B), 1:N_datasets)
            y_ndas = [Flux.onehotbatch((i+2)*ones(Int, B), 1:N_datasets) for i=1:N_nda]
            # y_ndas_π = [Flux.onehotbatch(2*ones(Int, B), 1:N_datasets) for i=1:N_nda]
            
            y = cat(y_demo, y_π, y_ndas..., dims=2) |> dev
            
            println("size y:$(size(y)), size D(x):$(size(D(x)))")
            # println(gradient_penalty(D, x_demo, x_π))
            # + 10f0 * gradient_penalty(D, x_demo, x_π)
            train!(D, (;kwargs...) -> Flux.Losses.logitcrossentropy(D(x), y), d_opt, info=info)
        end
        
        # ## replace the bufffer
        # rand!(𝒟_demo_batch, 𝒟_demo)
        # # rand!(𝒟_demo_π_batch, 𝒟_demo)
        # # 𝒟_demo_π_batch.data[:a] = action(π, 𝒟_demo_π_batch[:s])
        # rand!(𝒟_batch, 𝒮.buffer)
        # for i in 1:N_nda
        #     rand!(𝒟_ndas_batch[i], 𝒟_ndas[i])
        #     # rand!(𝒟_ndas_π_batch[i], 𝒟_ndas[i])
        #     # 𝒟_ndas_π_batch[i].data[:a] = action(π, 𝒟_ndas_π_batch[i][:s])
        # end
        # 
        # 𝒟_full = hcat(𝒟_demo_batch, 𝒟_batch, 𝒟_ndas_batch...)
        # 
        # for k in keys(𝒟)
        #     𝒟.data[k] = 𝒟_full.data[k]
        # end
        # 𝒟.elements = 𝒟_full.elements
        # 𝒟.next_ind = 𝒟_full.next_ind
        
        ## Compute the rewards
        D_out = Flux.softmax(value(D, flatten(𝒟[:s]), 𝒟[:a]))
        w = [1f0, 0f0, λ_nda*ones(Float32, N_nda)...] |> dev
        
        𝒟[:r] .= sum((Base.log.(D_out .+ 1f-5) .- Base.log.(1f0 .- D_out .+ 1f-5)) .* w, dims=1)
    end
    
    𝒮.post_batch_callback = GAIL_callback
    𝒮
end

