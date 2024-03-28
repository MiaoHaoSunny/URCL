from ast import arg
from statistics import mode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import os
from datasets.utils.ST_continual_dataset import generate_seq_dataset
from datasets.utils.load_PEMS08D4 import generate_seq_dataset_PEMS

from model.continual_model import ContinualModel
# from Baselines.Ablation_study.continual_model import ContinualModel
from model.STBuffer import Buffer
from data.get_logger import get_logger
from SeqDataset.seq_PEMSD408 import Sequential_PEMSD408
from GraphWaveNet import util
from datasets.utils.utils import simple_contrastive_loss

# def unified_train()


def train_PEMSD408(args):
    logger = get_logger(args.logfile)
    device = torch.device(args.device)
    buffer = Buffer(256, device=device)
    
    _, _, _, scaler = generate_seq_dataset_PEMS(file=args.data)
    bestid = 10000

    model = ContinualModel(args=args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criteria_loss = util.masked_mae
    
    # train
    # metr_la_dataset = Sequential_METR_LA(args=args)
    pemsD408_dataset = Sequential_PEMSD408(args=args)


    his_loss_list = []
    val_time_list = []
    train_time_list = []
    
    for task_id in range(pemsD408_dataset.N_TASKS):
        logger.info('Task {} begins.'.format(task_id))

        train_loader, val_loader, test_loader, _ = pemsD408_dataset.get_data_loaders(args)

        his_loss = []
        val_time = []
        train_time = []
        if task_id == 0:
            for i in range(1, args.epochs+1):
                train_loss = []
                train_mae = []
                train_mape = []
                train_rmse = []
                t1 = time.time()

                # train
                model.train()
                for iter, (x, y) in enumerate(train_loader):
                    trainx = x.float().to(device)
                    trainx = trainx.transpose(1, 3)
                    # print('origin trainx: {}'.format(trainx.shape))
                    trainy = y.float().to(device)
                    trainy = trainy.transpose(1, 3)
                    optimizer.zero_grad()
                    if buffer.is_empty():
                        lamda = None
                        trainx = F.pad(trainx, (1, 0, 0, 0))
                        # print('pad trainx: {}'.format(trainx.shape))
                        output, p1, p2, z1, z2 = model(trainx)
                        # output, p1, p2, z1, z2 = model(trainx)
                    else:
                        trainx = F.pad(trainx, (1, 0, 0, 0))
                        # print('pad trainx: {}'.format(trainx.shape))
                        buffer_x, buffer_y = buffer.get_mir_data(model=model, size=128, batch_size=args.batch_size, lr=args.learning_rate, args=args)
                        # buffer_x, buffer_y = buffer.get_data(size=args.batch_size)
                        lamda = np.random.beta(args.alpha, args.alpha)
                        output, p1, p2, z1, z2 = model(trainx, lamda=lamda, buffer_x=buffer_x)
                        # output, p1, p2, z1, z2 = model(trainx, lamda=lamda, buffer_x=buffer_x)
                    output = output.transpose(1, 3)
                    real = torch.unsqueeze(trainy[:, 0, :, :], dim=1)
                    real = scaler.inverse_transform(real)
                    predict = scaler.inverse_transform(output)

                    # p1, p2, z1, z2 = scaler.inverse_transform(p1), scaler.inverse_transform(p2), scaler.inverse_transform(z1), scaler.inverse_transform(z2)
                    
                    # if lamda is not None:
                    #     if predict.shape[0] != buffer_y.shape[0]:
                    #         id_list = predict.shape[0]
                    #         buffer_y = buffer_y[:id_list]
                    #     loss_mae = lamda * criteria_loss(predict, real, 0.0) + (1 - lamda) * criteria_loss(predict, buffer_y, 0.0)
                    # else:
                    #     loss_mae = criteria_loss(predict, real, 0.0)
                    loss_mae = criteria_loss(predict, real, 0.0)
                    loss_cl = 0.5 * simple_contrastive_loss(p1, z2) + 0.5 * simple_contrastive_loss(p2, z1)
                    # print(loss_cl)
                    loss = loss_mae + loss_cl
                    loss.backward()
                    if args.clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    mape = util.masked_mape(predict, real, 0.0).item()
                    rmse = util.masked_rmse(predict, real, 0.0).item()

                    train_loss.append(loss.item())
                    train_mae.append(loss_mae.item())
                    train_mape.append(mape)
                    train_rmse.append(rmse)

                    buffer.add_data(examples=trainx, logits=real)

                    if iter % args.print_every == 0:
                        log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                        logger.info(log.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]))
                t2 = time.time()
                train_time.append(t2-t1)

                # validation
                valid_mae = []
                valid_mape = []
                valid_rmse = []

                # print('NULL', valid_mae)

                s1 = time.time()
                model.eval()
                for iter, (x, y) in enumerate(val_loader):
                    testx = x.float().to(device)
                    testx = testx.transpose(1, 3)
                    testy = y.float().to(device)
                    testy = testy.transpose(1, 3)

                    testx = F.pad(testx, (1, 0, 0, 0))
                    output, _, _, _, _ = model(testx, aug1=None, aug2=None)
                    output = output.transpose(1, 3)
                    real = torch.unsqueeze(testy[:, 0, :, :], dim=1)
                    real = scaler.inverse_transform(real)
                    predict = scaler.inverse_transform(output)
                    mae = criteria_loss(predict, real, 0.0).item()
                    valid_mae.append(mae)
                    mape = util.masked_mape(predict, real, 0.0).item()
                    valid_mape.append(mape)
                    rmse = util.masked_rmse(predict, real, 0.0).item()
                    valid_rmse.append(rmse)
                s2 = time.time()
                val_time.append(s2-s1)
                mtrain_loss = np.mean(train_loss)
                mtrain_mae = np.mean(train_mae)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)

                # print('Full', valid_mae)

                mvalid_mae = np.mean(valid_mae)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                his_loss.append(mvalid_mae)
                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
                logger.info(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_mae, mvalid_mape, mvalid_rmse, (t2-t1)))
                torch.save(model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_mae,2))+'_task' + str(task_id)+".pth")
            logger.info('Average Training Time: {:.4f} sec/epoch'.format(np.mean(train_time)))
            logger.info('Average Inferecen Time: {:.4f} sec'.format(np.mean(val_time)))

            outputs = []
            realy = []

            model.eval()
            for iter, (x, y) in enumerate(test_loader):
                testx = x.float().to(device)
                testx = testx.transpose(1, 3)
                testy = y.float().to(device)
                testy = testy.transpose(1, 3)
                testy = testy[:, 0, :, :]
                with torch.no_grad():
                    preds, _, _, _, _ = model(testx)
                    preds = preds.transpose(1, 3)

                outputs.append(preds.squeeze())
                realy.append(testy.squeeze())
            
            yhat = torch.cat(outputs, dim=0)
            realy = torch.cat(realy, dim=0)

            logger.info('Task {} training finished'.format(task_id))
            logger.info('The valid loss on best model is {}'.format(round(his_loss[bestid], 4)))

            amae = []
            amape = []
            armse = []

            # print((realy[:, :, 0]==realy[:, :, 1]).all())
            # print(yhat.shape, realy.shape)
            for i in range(12):
                pred = scaler.inverse_transform(yhat[:, :, i])
                # inverse only used for PEMS04 or 08
                real = scaler.inverse_transform(realy[:, :, i])
                # real = realy[:, :, i]
                # print(pred.shape, real.shape)
                metrics = util.metric(pred, real)
                log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
                logger.info(log.format(i+1, metrics[0], metrics[1], metrics[2]))
                amae.append(metrics[0])
                amape.append(metrics[1])
                armse.append(metrics[2])

            his_loss_list.append(his_loss)
        else:
            model.load_state_dict(torch.load(args.save+'_epoch_'+str(bestid+1)+'_'+str(round(his_loss_list[task_id-1][bestid],2))+'_task' + str(task_id-1)+'.pth'))

            for i in range(1, args.epochs+1):
                train_loss = []
                train_mae = []
                train_mape = []
                train_rmse = []
                t1 = time.time()

                # train
                model.train()
                for iter, (x, y) in enumerate(train_loader):
                    trainx = x.float().to(device)
                    trainx = trainx.transpose(1, 3)
                    trainy = y.float().to(device)
                    trainy = trainy.transpose(1, 3)
                    optimizer.zero_grad()
                    # if buffer.is_empty():
                    #     trainx = F.pad(trainx, (1, 0, 0, 0))
                    #     output, p1, p2, z1, z2 = model(trainx)
                    # else:
                    #     trainx = F.pad(trainx, (1, 0, 0, 0))
                    #     buffer_x = buffer.get_mir_data(model=model, size=128, batch_size=args.batch_size, lr=args.learning_rate)
                    #     lamda = np.random.beta(args.alpha, args.alpha)
                    #     output, p1, p2, z1, z2 = model(trainx, lamda=lamda, buffer_x=buffer_x)
                    # output = output.transpose(1, 3)
                    # real = torch.unsqueeze(trainy[:, 0, :, :], dim=1)
                    # predict = scaler.inverse_transform(output)
                    # loss_mae = criteria_loss(predict, real, 0.0)
                    if buffer.is_empty():
                        lamda = None
                        trainx = F.pad(trainx, (1, 0, 0, 0))
                        # print('pad trainx: {}'.format(trainx.shape))
                        output, p1, p2, z1, z2 = model(trainx)
                        # output, p1, p2, z1, z2 = model(trainx)
                    else:
                        trainx = F.pad(trainx, (1, 0, 0, 0))
                        # print('pad trainx: {}'.format(trainx.shape))
                        buffer_x, buffer_y = buffer.get_mir_data(model=model, size=128, batch_size=args.batch_size, lr=args.learning_rate, args=args)
                        # buffer_x, buffer_y = buffer.get_data(size=args.batch_size)
                        lamda = np.random.beta(args.alpha, args.alpha)
                        output, p1, p2, z1, z2 = model(trainx, lamda=lamda, buffer_x=buffer_x)
                        # output, p1, p2, z1, z2 = model(trainx, lamda=lamda, buffer_x=buffer_x)
                    output = output.transpose(1, 3)
                    real = torch.unsqueeze(trainy[:, 0, :, :], dim=1)
                    real = scaler.inverse_transform(real)
                    predict = scaler.inverse_transform(output)

                    # p1, p2, z1, z2 = scaler.inverse_transform(p1), scaler.inverse_transform(p2), scaler.inverse_transform(z1), scaler.inverse_transform(z2)
                    
                    # if lamda is not None:
                    #     if predict.shape[0] != buffer_y.shape[0]:
                    #         id_list = predict.shape[0]
                    #         buffer_y = buffer_y[:id_list]
                    #     loss_mae = lamda * criteria_loss(predict, real, 0.0) + (1 - lamda) * criteria_loss(predict, buffer_y, 0.0)
                    # else:
                    #     loss_mae = criteria_loss(predict, real, 0.0)
                    loss_mae = criteria_loss(predict, real, 0.0)

                    loss_cl = 0.5 * simple_contrastive_loss(p1, z2) + 0.5 * simple_contrastive_loss(p2, z1)
                    loss = loss_mae + loss_cl
                    loss.backward()
                    if args.clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    mape = util.masked_mape(predict, real, 0.0).item()
                    rmse = util.masked_rmse(predict, real, 0.0).item()

                    train_loss.append(loss.item())
                    train_mae.append(loss_mae.item())
                    train_mape.append(mape)
                    train_rmse.append(rmse)

                    buffer.add_data(examples=trainx, logits=real)

                    if iter % args.print_every == 0:
                        log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                        logger.info(log.format(iter, train_loss[-1], train_mae[-1], train_mape[-1], train_rmse[-1]))
                t2 = time.time()
                train_time.append(t2-t1)

                # validation
                valid_mae = []
                valid_mape = []
                valid_rmse = []

                s1 = time.time()
                model.eval()
                for iter, (x, y) in enumerate(val_loader):
                    testx = x.float().to(device)
                    testx = testx.transpose(1, 3)
                    testy = y.float().to(device)
                    testy = testy.transpose(1, 3)

                    testx = F.pad(testx, (1, 0, 0, 0))
                    output, _, _, _, _ = model(testx, aug1=None, aug2=None)
                    output = output.transpose(1, 3)
                    real = torch.unsqueeze(testy[:, 0, :, :], dim=1)
                    real = scaler.inverse_transform(real)
                    predict = scaler.inverse_transform(output)
                    mae = criteria_loss(predict, real, 0.0).item()
                    valid_mae.append(mae)
                    mape = util.masked_mape(predict, real, 0.0).item()
                    valid_mape.append(mape)
                    rmse = util.masked_rmse(predict, real, 0.0).item()
                    valid_rmse.append(rmse)
                s2 = time.time()
                val_time.append(s2-s1)
                mtrain_loss = np.mean(train_loss)
                mtrain_mae = np.mean(train_mae)
                mtrain_mape = np.mean(train_mape)
                mtrain_rmse = np.mean(train_rmse)

                mvalid_mae = np.mean(valid_mae)
                mvalid_mape = np.mean(valid_mape)
                mvalid_rmse = np.mean(valid_rmse)
                his_loss.append(mvalid_mae)
                log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f} Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
                logger.info(log.format(i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_mae, mvalid_mape, mvalid_rmse, (t2-t1)))
                torch.save(model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_mae,2))+'_task' + str(task_id)+".pth")
            logger.info('Average Training Time: {:.4f} sec/epoch'.format(np.mean(train_time)))
            logger.info('Average Inferecen Time: {:.4f} sec'.format(np.mean(val_time)))

            # testing
            bestid = np.argmin(his_loss)
            model.load_state_dict(torch.load(args.save+'_epoch_'+str(bestid+1)+'_'+str(round(his_loss[bestid],2))+'_task' + str(task_id)+'.pth'))

            outputs = []
            realy = []

            model.eval()
            for iter, (x, y) in enumerate(test_loader):
                testx = x.float().to(device)
                testx = testx.transpose(1, 3)
                testy = y.float().to(device)
                testy = testy.transpose(1, 3)
                testy = testy[:, 0, :, :]
                with torch.no_grad():
                    preds, _, _, _, _ = model(testx)
                    preds = preds.transpose(1, 3)

                outputs.append(preds.squeeze())
                realy.append(testy.squeeze())
            
            yhat = torch.cat(outputs, dim=0)
            realy = torch.cat(realy, dim=0)

            logger.info('Task {} training finished'.format(task_id))
            logger.info('The valid loss on best model is {}'.format(round(his_loss[bestid], 4)))

            amae = []
            amape = []
            armse = []

            # print((realy[:, :, 0]==realy[:, :, 1]).all())
            # print(yhat.shape, realy.shape)
            for i in range(12):
                pred = scaler.inverse_transform(yhat[:, :, i])
                real = scaler.inverse_transform(realy[:, :, i])
                # real = realy[:, :, i]
                # print(pred.shape, real.shape)
                metrics = util.metric(pred, real)
                log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
                logger.info(log.format(i+1, metrics[0], metrics[1], metrics[2]))
                amae.append(metrics[0])
                amape.append(metrics[1])
                armse.append(metrics[2])

            his_loss_list.append(his_loss)
