function [w, run_time,Loss] = OMVFS(X, option)
    % Multi view feature selection
    % X cell array, contains multi view data. each matrix is N by d_v
    % option.block_size
    % option.buffer_size
    % option.beta vector of size n_v
    % option.num_cluster
    % option.pass
    % option.maxiter
    % option.tol
    num_passes = option.pass;
    max_buffer_size = option.buffer_size;
    beta = option.beta;
    alpha = option.alpha;
    lambda = option.lambda;
    maxIter = option.maxiter;
    num_clusters = option.num_cluster;
    block_size = option.block_size;
    tol = option.tol;
    gamma = option.gamma;
    num_views = numel(X);
    num_feature = zeros(num_views,1);
    total = size(X{1},1);
    for i = 1:num_views
        num_feature(i) = size(X{i},2);
    end

    current_buffer_size = 0;

    % Initialize U and V
    U_buff = [];
    V = cell(num_views,1);
    for i = 1:num_views
        V{i} = rand(num_feature(i),num_clusters);
    end

    A = cell(num_views,1);
    B = cell(num_views,2);
    for i =1:num_views
        A{i} = zeros(size(V{i}));
        B{i} = zeros(num_clusters, num_clusters);
    end

    num_block = ceil(total/block_size);
    Loss = zeros(num_passes, num_block);

    index_sofa = 0;
    U_sofar = cell(num_views,1);
    for i = 1:num_views
        U_sofa{i} = [];
    end
    for pass = 1:num_passes
        X_buff = cell(num_views,1);
        for i = 1:num_views
            X_buff{i} = [];
        end
        for block_index = 1:num_block
            data_range = (block_index-1)*block_size+1:block_index*block_size;
            if block_index == num_block
                data_range = (block_index-1)*block_size+1:total;
            end

            X_i = cell(num_views,1);

            D = cell(num_views, 1);
            L = cell(num_views, 1);
            DD = cell(num_views, 1);
            SS = cell(num_views, 1);
            for i = 1:num_views
                if size(X_buff{i},1) + numel(data_range) <= max_buffer_size
                    X_buff{i} = [X_buff{i}; X{i}(data_range,:)];
                else
                    X_buff{i} = [X_buff{i}; X{i}(data_range,:)];
                    X_buff{i} = X_buff{i}(end- max_buffer_size+1:end,:);
                end
                X_i{i} = X{i}(data_range,:);
                [L{i}, DD{i}, SS{i}] = construct_Laplacian(X_buff{i});
                D{i} = diag(sparse(ones(1,num_feature(i))));
            end

            for i = 1:num_views
                U_buff{i} = rand(size(X_buff{i},1), num_clusters);
                U_i{i} = U_buff{i}(end-numel(data_range)+1:end,:);
                if size( U_buff{i},1 ) > max_buffer_size
                    U_buff{i} = U_buff{i}(end- max_buffer_size+1:end,:);
                end
                U_sofar{i} = [U_sofar{i}; U_i{i}];
            end
            U_star_buff = rand(size(U_buff{1}));
            index_sofa = index_sofa + numel(data_range);


            M = alpha(1)*L{1};
            for ii = 2:num_views
                M = M + alpha(ii)*L{ii};
            end
            M_plus = 0.5*(abs(M) + M);
            M_minus = 0.5*(abs(M) - M);

            iter = 0;
            converge = 0;
            step_u = 0.5;
            step_v = ones(num_views,1);
            beta_search = 0.1;
            while iter < maxIter && converge ==0
                % update U_star_buff
                U_star_buff = lambda(1)*U_buff{1};
                for i = 2:num_views
                    U_star_buff = U_star_buff + lambda(i)*U_buff{2};
                end
                U_star_buff = U_star_buff/sum(lambda);

                % Update U_i
                for i = 1:num_views
                    upper_tmp = X_buff{i}*V{i} + U_star_buff + gamma*U_buff;
                    lower_tmp = U_buff{i}*(V{i}'*V{i}) + alpha(i)*L{i}*U_buff{i} + U_buff{i}+ gamma*U_buff*(U_buff'*U_buff);
                    U_buff{i} = U_buff{i}.*(upper_tmp./max(lower_tmp,1e-20)).^step_u;
                end
                U_i{i} = U_buff(end-numel(data_range)+1:end,:);

                % Update V_i
                for i = 1:num_views
                    upper_tmp = A{i} + X_i{i}' * U_i{i};
                    lower_tmp = V{i} * (B{i} + (U_i{i}' * U_i{i}));
                    % construct D
                    D{i} = diag(sparse(0.5./max(sqrt(sum(V{i}.^2,2)),1e-20)));
                    lower_tmp = lower_tmp + beta(i) * D{i} * V{i};
                    V{i} = V{i} .* (upper_tmp./max(lower_tmp, 1e-20)).^step_u;
                end

                for i = 1:num_views
                    U_sofar{i}(data_range,:) = U_i{i};
                end
                iter = iter + 1;
            end
            log = 0;
            % Update A and B
            for i = 1:num_views
                A{i} = A{i} + X_i{i}' * U_i{i};
                B{i} = B{i} + U_i{i}' * U_i{i};
            end
            fprintf('Done with block %d in %d iterations and log is %d\n', block_index, iter, log);
        end
        fprintf('Done with pass %d\n', pass);
    end
    for i = 1:num_views
        w{i} = sqrt(sum(V{i}.^2,2));
    end
end

function [result] = L21(X)
    % L2,1 norm of X, the sum of row norms
    result = sum(sqrt(sum(X.^2,2)));
end

function [L,V,S] = construct_Laplacian(X)
    n=size(X,1);
    S = kernelmatrix(X,1);
    V = zeros(size(S));
    for i = 1:size(V,1)
        V(i,i) = sum(S(:,i));
    end
    L = V - S;
end

function K = kernelmatrix(coord, sig)
    n=size(coord,1);
    K=coord*coord'/sig^2;
    d=diag(K);
    K=K-ones(n,1)*d'/2;
    K=K-d*ones(1,n)/2;
    K=exp(K);
end
