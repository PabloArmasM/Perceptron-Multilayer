function []=main ()

    %1000 Fotos.
    
    ejecutando = 'ejecutando'
    alpha=0.95
    
    generaciones = 1000;
    ng = 0;
    
    %w1 =rand(784,784);
    w2 = rand(784, 10);
    w3 = rand(10,1);
    
    %u1=rand(1,784);
    %u2=rand(1,784);
    u2=rand(1,10);
    u3=rand;
    
    Out=[];
    AG = [];
    MSEG = [];
    
    matrizEntrada = loadMNISTImages('entrenamiento1');
    matrizRespuesta = loadMNISTLabels('entrenamiento2_Labels');
    while generaciones > ng
        for i=1:1:size(matrizEntrada,2)
            imagen= [matrizEntrada(:,i)'];
            
            %A1 = 1./(1+exp(-1*(imagen*w1+u1)./100));
            %A2 = 1./(1+exp(-1*(A1*w2+u2)./100));
            %out = 1./(1+exp(-1*(A2*w3+u3)./100));
            %A1 = sigmf(((imagen*w1+u1)./100),[1,0]);
            A2 = sigmf(((imagen*w2+u2)./100),[1,0]);
            out = sigmf(((A2*w3+u3)./100),[1,0]);
            Out = [Out,out];
            
            esp = matrizRespuesta(i);
            DeltaOut = out.*(1-out).*(matrizRespuesta(i)/10-out);
            Delta2 = CalculaDelta(A2, DeltaOut, w3);
            %Delta1 = CalculaDelta(A1,Delta2,w2);
            
            
            %aux = alpha*imagen'*Delta1;
            %w1 = w1 + (alpha*imagen'*Delta1);
            w2 = w2 + (alpha*imagen'*Delta2);
            w3 = w3 + (alpha*A2'*DeltaOut);
            
            %u1 = u1 + (alpha * Delta1);
            u2 = u2 + (alpha * Delta2);
            u3 = u3 + (alpha * DeltaOut);
        end
        pS = 0;
        Aciertos = 0;
        for i=1:1:length(Out)
           pS = pS + ((matrizRespuesta(i)-(Out(i)*10))^2);
           if matrizRespuesta(i) == round(Out(i)*10)
                Aciertos = Aciertos + 1;
           end
        end
        MSE = ((1/length(Out))*pS)
        AciertosTotal = (1/length(Out))*Aciertos
        AG = [AG, AciertosTotal];
        MSEG = [MSEG, MSE];
        if AciertosTotal >= 0.9
           break; 
        end
        Out = [];
        ng=ng+1;
    end
    matrizTest = loadMNISTImages('conjunto1');
    matrizTestLabel = loadMNISTLabels('conjunto2_labels');
    aciertos = 0;
    for i=1:1:size(matrizTest,2)
       %A1 = sigmf(((matrizTest(:,i)'*w1+u1)./100),[1,0]);
       A2 = sigmf(((matrizTest(:,i)'*w2+u2)./100),[1,0]);
       out = sigmf(((A2*w3+u3)./100),[1,0]); 
       if matrizTestLabel(i) == round(out*10) 
            aciertos = aciertos + 1;
       end
    end
    aciertosTotales = aciertos/length(matrizTestLabel)
    aciertos
    subplot(2,2,1);
    plot(AG);
    title('Aciertos totales');
    subplot(2,2,2);
    plot(MSEG);
    title('Error');
end