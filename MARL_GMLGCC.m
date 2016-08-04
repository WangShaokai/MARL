function [ Pz_d, logLs, Pz_ds, accs ] = MARL_GMLGCC( Qz_d, fea, link, testIdx, trainIdx, alpha, numK)

    [numlabel, N] = size(Qz_d);
    AllResults = [];
    triID = 1;
    ZERO_OFFSET = 1e-200;
    num_Graph = length(fea);
    for j=1:num_Graph
        fea{j} = fea{j}';
    end
    fea_2 = cat(1,fea{:});
    TrainY = zeros(1,N);
    for i=1:N
        TrainY(i)=find(Qz_d(:,i)==1);
    end       

%     Pz_d = rand(numlabel,N);
%     for id = 1:length(TrainY)
%         Pz_d(:,id) = Pz_d(:,id)/sum(Pz_d(:,id));
%     end    
%     Pz_d(:,trainIdx) = Qz_d(:,trainIdx);
    TestY = TrainY(testIdx);
    
    trainIdx2 = trainIdx;
    [~, label_train] = max(Qz_d(:,trainIdx)', [], 2);
    diff = setdiff(1:numlabel, unique(label_train));
    label_vector = zeros(length(diff),numlabel);
    a1 = zeros(length(diff),1);
    for i=1:length(diff)
        label_vector(i,diff(i)) = 1;
        a = find(ismember(Qz_d',label_vector(i,:),'rows'));
        a1(i) = a(1);
    end
    trainIdx2(a1)=1;
    
        % Training Naive Bayes models
        Pz_d_v = cell(1, num_Graph);
        for j=1:num_Graph
            [pc pw_c] = trainNBC(fea{j}(:,trainIdx2), TrainY(trainIdx2));
            [predictMatrix Auc] = testNBC(pc,pw_c,fea{j}(:,testIdx),TrainY(testIdx));
            fprintf('The accuracy on Test data set by view1 is %g....\n',Auc)
            AllResults(triID,2) = Auc;        

            Pz_d(:,testIdx) = predictMatrix;
            % deal with NaN
            [B C] = find(isnan(Pz_d));
            for id = 1:length(B)
                Pz_d(:,C(id)) = ones(numlabel,1)/numlabel;
            end
            Pz_d_v{j} = Pz_d;   
        end            
        Pz_d =Pz_d_v{1};  
        for j=2:num_Graph
            Pz_d = Pz_d + Pz_d_v{j};
        end        
        Pz_d = Pz_d/num_Graph;
        Pz_d(:,trainIdx) = Qz_d(:,trainIdx);        

        Pw_y = cell(1, num_Graph);
        Py_z = cell(1, num_Graph);
        for j=1:num_Graph
             Pw_y{j} = rand(size(fea{j},1),numK);
            for iy = 1:size(Pw_y{j},2)
                Pw_y{j}(:,iy) = Pw_y{j}(:,iy)/sum(Pw_y{j}(:,iy));
            end           
            Py_z{j} = rand(numK,numlabel);
            for iz = 1:numlabel
                Py_z{j}(:,iz) = Py_z{j}(:,iz)/sum(Py_z{j}(:,iz));
            end   
        end                
        
        A = Pz_d(:,testIdx);
        nCorrect = 0;
        for did = 1:length(TestY)
            [C D] = max(A(:,did));
            if D(1) == TestY(did)
                nCorrect = nCorrect + 1;
            end
        end
        fprintf('In Iteration %g, the accuracy on Test data set is %g....\n',0,nCorrect*100/length(TestY));  
        
    DCol = full(sum(link, 2));
    DCol(DCol>0) = DCol(DCol>0).^(-1/2);
    D = spdiags(DCol, 0, N, N);
    mid = D*link*D;
    L = speye(N, N) - mid;
    L = alpha*L;
    dLen = full(sum(fea_2,1));
    Omega = spdiags(dLen', 0, N, N);
    OmegaL = Omega + L;           
    
    Pd = cell(1, num_Graph);
    Pw_z = cell(1, num_Graph);
    Pw_d = cell(1, num_Graph);
    for j=1:num_Graph
        Pd{j} = sum(fea{j})./sum(fea{j}(:));
        Pd{j} = full(Pd{j});
        Pw_z{j} = Pw_y{j}*Py_z{j};
        Pw_d{j} = mex_Pw_d(fea{j},Pw_z{j},Pz_d);
    end    
    maxIter = 100;
    logLs = zeros(maxIter,1);
    accs = zeros(maxIter,1);
    Pz_ds = zeros(numlabel, N, maxIter);
    % EM
    logL =0;  
    for j=1:num_Graph
        logL = logL + mex_logL(fea{j},Pw_d{j},Pd{j});
    end    
    logL = logL - sum(sum((log(Pz_d + ZERO_OFFSET)*L).*Pz_d));%55555555555555
    pre_Pz_d = Pz_d;
    pre_logL = logL;    
    
    for iter = 1:maxIter
        temp = cell(1, num_Graph);
        temp_Pw_y = cell(1, num_Graph);
        temp_Py_z = cell(1, num_Graph);    
        for j=1:num_Graph
            temp{j} = Pw_y{j}*Py_z{j}*Pz_d;
            temp{j}(find(temp{j} == 0)) = 1;
            temp{j} = fea{j}./temp{j};  
            temp_Pw_y{j} = Pw_y{j}.*(temp{j}*Pz_d'*Py_z{j}');
            for yid = 1:size(Pw_y{j},2)
                temp_Pw_y{j}(:,yid) = temp_Pw_y{j}(:,yid)/sum(temp_Pw_y{j}(:,yid));
            end
            temp_Py_z{j} = Py_z{j}.*(Pw_y{j}'*temp{j}*Pz_d');
            for zid = 1:size(Py_z{j},2)
                temp_Py_z{j}(:,zid) = temp_Py_z{j}(:,zid)/sum(temp_Py_z{j}(:,zid));
            end
        end

        Pw_y_z = Py_z{1}'*Pw_y{1}'*temp{1};
        for j=2:num_Graph
            Pw_y_z = Pw_y_z+Py_z{j}'*Pw_y{j}'*temp{j};
        end                                 
            Pz_d = Pz_d.*Pw_y_z; 
            Pz_d = (OmegaL\Pz_d')';           %5555555555555555555555
            for did = 1:size(Pz_d,2)
                Pz_d(:,did) = Pz_d(:,did)/sum(Pz_d(:,did));
            end
        Pz_d(:,trainIdx) = Qz_d(:,trainIdx);
        for j=1:num_Graph
            Pw_y{j} = temp_Pw_y{j};
            Py_z{j} = temp_Py_z{j};
            Pw_z{j} = Pw_y{j}*Py_z{j};
            Pw_d{j} = mex_Pw_d(fea{j},Pw_z{j},Pz_d);
        end                    

        A = Pz_d(:,testIdx);            
        nCorrect = 0;
        for did = 1:length(TestY)
            [C D] = max(A(:,did));
            if D(1) == TestY(did)
                nCorrect = nCorrect + 1;
            end
        end  
        logL =0;  
        for j=1:num_Graph
            logL = logL + mex_logL(fea{j},Pw_d{j},Pd{j});
        end                 
        logL = logL - sum(sum((log(Pz_d + ZERO_OFFSET)*L).*Pz_d));  %55555555555555555555
        logLs(iter) = logL;
        accs(iter) = nCorrect*100/length(TestY);
        Pz_ds(:,:,iter) = Pz_d;
        delta = (pre_logL - logL)/pre_logL;
        disp(['iter:', num2str(iter), ' logL:', num2str(logL), ' delta:', num2str(delta), ' the accuracy on Test data set is:', num2str(nCorrect*100/length(TestY))]);
        if delta < 1e-4
            if delta < 0
                Pz_d = pre_Pz_d;
                iter = iter - 1;
            end
            logLs = logLs(1:iter);
            accs = accs(1:iter);
            Pz_ds = Pz_ds(:,:,1:iter);
            break;
        else
            pre_Pz_d = Pz_d;
            pre_logL = logL;
        end            
            
    end
end

