simulated_cfs = None
if simulated_cfs:
    encoder.decode(train_df)

    long_data = pd.wide_to_long(train_df, stubnames=cols, i='trace_id',
                                j='order', sep='_', suffix=r'\w+')
    long_data_sorted = long_data.sort_values(['trace_id', 'order'], ).reset_index(drop=False)
    long_data_sorted.drop(columns=['order'], inplace=True)
    columns_to_rename = {'trace_id': 'case:concept:name'}
    columns_to_rename.update({'prefix': 'concept:name'})
    long_data_sorted.rename(columns=columns_to_rename, inplace=True)
    long_data_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
    long_data_sorted.replace('0', 'other', inplace=True)
    train_log = convert_to_event_log(long_data_sorted)
    import ast

    if 'sepsis' in simulated_cfs:
        if simulated_cfs.endswith('.xes'):
            simulated_log = get_log(filepath=simulated_cfs)
            simulated_log = pm4py.convert_to_dataframe(simulated_log)
            dicts = {}
            for i in range(len(simulated_log)):
                dicts[i] = ast.literal_eval(simulated_log.loc[i][-1])
            df = pd.DataFrame.from_dict(dicts, orient='index')
            simulated_log = pd.merge(simulated_log, df, how='inner', on=df.index)
            simulated_log.drop(columns=['key_0', 'amount', 'st_wip',
                                        'st_tsk_wip', 'queue', 'arrive:timestamp'], inplace=True)
            simulated_log.rename(
                columns={'role': 'org:resource', 'task': 'concept:name', 'caseid': 'case:concept:name'},
                inplace=True)
            simulated_log['org:group'] = simulated_log['org:resource']
            simulated_log['label'] = 'true'
            simulated_log['lifecycle:transition'] = 'complete'
        else:
            simulated_log = pd.read_csv(simulated_cfs)
            dicts = {}
            for i in range(len(simulated_log)):
                dicts[i] = ast.literal_eval(simulated_log.loc[i][-1])
            df = pd.DataFrame.from_dict(dicts, orient='index')
            simulated_log = pd.merge(simulated_log, df, how='inner', on=df.index)
            simulated_log.drop(columns=['key_0', 'amount', 'st_wip',
                                        'st_tsk_wip', 'queue', 'arrive:timestamp'], inplace=True)
            simulated_log.rename(
                columns={'role': 'org:resource', 'task': 'concept:name', 'caseid': 'case:concept:name'}, inplace=True)
            simulated_log['org:group'] = simulated_log['org:resource']
            simulated_log['label'] = 'true'
            simulated_log['lifecycle:transition'] = 'complete'

    train_log = pm4py.convert_to_dataframe(train_log)
    if '2012' in simulated_log:
        simulated_log = simulated_log.rename(columns={'amount': 'AMOUNT_REQ', 'role': 'org:resource'})
        simulated_log['lifecycle:transition'] = 'complete'
        simulated_log['label'] = 'true'

    simulated_log = simulated_log[train_log.columns]
    simulated_log = pm4py.convert_to_event_log(simulated_log)

    if '2012' in dataset_name:
        for i in range(len(simulated_log)):
            for x in range(len(cols)):
                simulated_log[i].attributes.update({cols[x]: simulated_log[i][0]._dict[cols[x]]})
            for j in range(len(simulated_log[i])):
                for x in range(len(cols)):
                    del simulated_log[i][j]._dict[cols[x]]
    elif 'sepsis' in dataset_name:
        cols = [*dataset_confs.static_cat_cols.values(), *dataset_confs.static_num_cols.values()]
        cols = list(itertools.chain.from_iterable(cols))
        cols.append('label')
        for i in range(len(simulated_log)):
            for x in range(len(cols)):
                simulated_log[i].attributes.update({cols[x]: simulated_log[i][0]._dict[cols[x]]})
            for j in range(len(simulated_log[i])):
                for x in range(len(cols)):
                    del simulated_log[i][j]._dict[cols[x]]
    encoder, simulated_df = get_encoded_df(log=simulated_log, encoder=encoder, CONF=CONF)
    predicted_simulated = best_model.model.predict(drop_columns(simulated_df))
    simulated_df['label'] = predicted_simulated
    updated_train_df = simulated_df
