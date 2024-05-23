from datetime import timedelta
import simpy
import pm4py
import random

from rims.checking_process import SimulationProcess
from pm4py.objects.petri_net import semantics
from rims.MAINparameters import*

ATTRIBUTES = {
        'sepsis_cases_1_start': {'TRACE': ['Age', 'Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG', 'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                 'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture', 'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg', 'Hypotensie',
                 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie', 'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea', 'SIRSCritTemperature', 'SIRSCriteria2OrMore'], 'EVENT': ['CRP', 'LacticAcid', 'Leucocytes', 'event_nr', 'hour', 'month', 'timesincecasestart', 'timesincelastevent', 'timesincemidnight', 'weekday']},
        'BPI_Challenge_2012_W_Two_TS':{'TRACE': ['AMOUNT_REQ'], 'EVENT': []},
        'bpic2015_2_start': {'TRACE': ['Aanleg (Uitvoeren werk of werkzaamheid)', 'Bouw',
                                             'Brandveilig gebruik (melding)', 'Brandveilig gebruik (vergunning)',
                                             'Gebiedsbescherming', 'Handelen in strijd met regels RO',
                                             'Inrit/Uitweg', 'Kap', 'Milieu (melding)',
                                             'Milieu (neutraal wijziging)',
                                             'Milieu (omgevingsvergunning beperkte milieutoets)',
                                             'Milieu (vergunning)', 'Monument', 'Reclame', 'Responsible_actor',
                                             'SUMleges', 'Sloop'], 'EVENT': ['event_nr', 'hour','lifecycle:transition', 'month',
                                                                             'question', 'timesincecasestart',
                                                                             'timesincelastevent', 'timesincemidnight',
                                                                                 'weekday']},
        'sepsis_cases_2_start': {
          'TRACE': ['Age', 'Diagnose', 'DiagnosticArtAstrup','DiagnosticBlood','DiagnosticECG',
             'DiagnosticIC','DiagnosticLacticAcid','DiagnosticLiquor','DiagnosticOther',
             'DiagnosticSputum','DiagnosticUrinaryCulture','DiagnosticUrinarySediment',
             'DiagnosticXthorax','DisfuncOrg','Hypotensie','Hypoxie',
             'InfectionSuspected','Infusion','Oligurie','SIRSCritHeartRate',
             'SIRSCritLeucos','SIRSCritTachypnea','SIRSCritTemperature','SIRSCriteria2OrMore'],
           'EVENT': ['CRP', 'LacticAcid', 'Leucocytes', 'event_nr', 'hour', 'month',
                                'timesincecasestart',
                                'timesincelastevent', 'timesincemidnight', 'weekday']},
        'sepsis_cases_3_start': {
                  'TRACE': ['Age', 'Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                            'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                            'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                            'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg', 'Hypotensie',
                            'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie', 'SIRSCritHeartRate',
                            'SIRSCritLeucos', 'SIRSCritTachypnea', 'SIRSCritTemperature', 'SIRSCriteria2OrMore'],
                  'EVENT': ['CRP', 'LacticAcid', 'Leucocytes', 'event_nr', 'hour', 'month', 'timesincecasestart',
                            'timesincelastevent', 'timesincemidnight', 'weekday']},
        'bpic2015_4_start': {'TRACE': ['Aanleg (Uitvoeren werk of werkzaamheid)', 'Bouw','Brandveilig gebruik (vergunning)',
                                                             'Gebiedsbescherming', 'Handelen in strijd met regels RO',
                                                             'Inrit/Uitweg', 'Kap',
                                                             'Milieu (neutraal wijziging)',
                                                             'Milieu (omgevingsvergunning beperkte milieutoets)',
                                                             'Milieu (vergunning)', 'Monument', 'Reclame', 'Responsible_actor',
                                                             'SUMleges', 'Sloop'],
                             'EVENT': ['event_nr', 'hour','lifecycle:transition', 'month',
                                       'question', 'timesincecasestart','timesincelastevent', 'timesincemidnight',
                                                                                                 'weekday']},
        'bpic2012_2_start': {'TRACE': ['AMOUNT_REQ'],'EVENT': ["hour", "weekday", "month", "timesincemidnight",
                                                                    "timesincelastevent",
                                                                    "timesincecasestart", "event_nr"]},
    'bpic2012_2_start_old': {'TRACE': ['AMOUNT_REQ'],'EVENT': ["hour", "weekday", "month", "timesincemidnight",
                                                                    "timesincelastevent",
                                                                    "timesincecasestart", "event_nr"]},
    'Productions': {'TRACE': ["Part_Desc_", "Rework", "Report_Type",
                                              "Work_Order_Qty"],
                        'EVENT': ["Qty_Completed", "Qty_for_MRB", "activity_duration", "event_nr",
                        "hour", "lifecycle:transition", "month", "timesincecasestart", "timesincelastevent",
                                  "timesincemidnight", "weekday"]}}


class Token(object):

    def __init__(self, id, start, params, process: SimulationProcess, sequence, contrafactual, NAME_EXPERIMENT, rp_feature='all_role'):
        self.id = id
        self.net, self.am, self.fm = pm4py.read_pnml(params.PATH_PETRINET)
        self.process = process
        self.start_time = start
        self.pr_wip_initial = params.PR_WIP_INITIAL
        self.rp_feature = rp_feature
        self.params = params
        self.see_activity = False
        self.pos = 0
        self.prefix = []
        self.sequence = sequence
        self.contrafactual = contrafactual
        self.CF = True if self.contrafactual else False
        self.NAME_EXPERIMENT = NAME_EXPERIMENT

    def read_json(self, path):
        with open(path) as file:
            data = json.load(file)
            self.ac_index = data['ac_index']

    def simulation(self, env: simpy.Environment, writer, type, syn=False):
        #trans = self.next_transition(syn)
        event = self.next_event() if self.contrafactual == False else self.next_event_contrafactual()
        ### register trace in process ###
        resource_trace = self.process.get_resource_trace()
        resource_trace_request = resource_trace.request()
        time_previous_event = self.start_time
        while event is not None:
            yield resource_trace_request
            buffer = [self.id, event[0]]
            buffer.append(str(self.start_time + timedelta(seconds=env.now))[:19])
            ### call predictor for waiting time
            if str(event[2]) != 0 and str(event[2]) in self.params.RESOURCE_TO_ROLE_LSTM:
                role = self.params.RESOURCE_ROLE[str(event[2])]
                resource = self.process.get_single_resource(str(event[2]))
            else:
                if self.NAME_EXPERIMENT == 'bpic2015_2_start':
                    role = self.params.RESOURCE_ROLE["560532"]
                    resource = self.process.get_single_resource("560532")  ## ruolo
                elif self.NAME_EXPERIMENT == 'bpic2015_4_start':
                    role = self.params.RESOURCE_ROLE["560752"]
                    resource = self.process.get_single_resource("560532")  ## ruolo
                elif self.NAME_EXPERIMENT == 'sepsis_cases_1_start' or self.NAME_EXPERIMENT == 'sepsis_cases_2_start' or self.NAME_EXPERIMENT == 'sepsis_cases_3_start':
                    role = self.params.RESOURCE_ROLE["F"]
                    resource = self.process.get_single_resource("F")
                elif self.NAME_EXPERIMENT == 'BPI_Challenge_2012_W_Two_TS':
                    role = self.params.RESOURCE_ROLE["11169.0"]
                    resource = self.process.get_single_resource("11169.0")
                elif self.NAME_EXPERIMENT == 'Productions':
                    role = self.params.RESOURCE_ROLE["ID4932"]
                    resource = self.process.get_single_resource("ID4932")

            if event[0] not in self.params.INDEX_AC:
                if self.NAME_EXPERIMENT == 'bpic2015_2_start' or self.NAME_EXPERIMENT == 'bpic2015_4_start':
                    event[0] = "05_EIND_010"
            transition = (self.params.INDEX_AC[event[0]], self.params.INDEX_ROLE[role])
            self.prefix.append(event[0])
            pr_wip_wait = self.pr_wip_initial + resource_trace.count
            #rp_oc = self.process.get_occupations_resource(role)
            rp_oc = self.process.get_occupations_all_role(role)
            request_resource = resource.request_CF() if self.contrafactual else resource.request_original()
            if len(resource.queue) > 0:
                queue = len(resource.queue[-1])
            else:
                queue = 0
            if self.NAME_EXPERIMENT == 'BPI_Challenge_2012_W_Two_TS':
                waiting = self.process.get_predict_waiting(str(self.id), pr_wip_wait, transition, rp_oc,
                                                           self.start_time + timedelta(seconds=env.now), queue)
            else:
                waiting = self.process.get_predict_waiting(str(self.id), pr_wip_wait, transition, rp_oc,
                                                       self.start_time + timedelta(seconds=env.now), -1)
            if self.contrafactual is not None:
                if self.see_activity:
                    yield env.timeout(waiting)
            else:
                yield env.timeout(float(event[3]))

            #### event attributes
            ### SEPSIS_ATTRIB_EVENT = ['CRP', 'LacticAcid', 'Leucocytes', 'event_nr', 'hour', 'month', 'timesincecasestart', 'timesincelastevent', 'timesincemidnight', 'weekday']
            attrib = []
            for a in ATTRIBUTES[self.NAME_EXPERIMENT]['EVENT']:
                if a == 'event_nr':
                    attrib.append(len(self.prefix))
                elif a == 'hour':
                    attrib.append((self.start_time + timedelta(seconds=env.now)).hour)
                elif a == 'month':
                    attrib.append((self.start_time + timedelta(seconds=env.now)).month)
                elif a == 'timesincecasestart':
                    attrib.append((((self.start_time + timedelta(seconds=env.now)) - self.start_time).total_seconds())/60)
                elif a == 'timesincelastevent':
                    attrib.append((((self.start_time + timedelta(seconds=env.now) - time_previous_event)).total_seconds())/60)
                elif a == 'timesincemidnight':
                    now = self.start_time + timedelta(seconds=env.now)
                    attrib.append((now.hour * 3600 + now.minute * 60 + now.second)/60)
                elif a == 'weekday':
                    attrib.append((self.start_time + timedelta(seconds=env.now)).isoweekday())
                else:
                    attrib.append(event[-3][a])  # event attributes
            time_previous_event = self.start_time + timedelta(seconds=env.now)

            yield request_resource
            ### register event in process ###
            resource_task = self.process.get_resource_event(event[0])
            resource_task_request = resource_task.request()
            yield resource_task_request
            buffer.append(str(self.start_time + timedelta(seconds=env.now))[:19])
            ### call predictor for processing time
            pr_wip = self.pr_wip_initial + resource_trace.count
            #rp_oc = self.process.get_occupations_resource(resource.get_name())
            rp_oc = self.process.get_occupations_all_role(role)
            initial_ac_wip = self.params.AC_WIP_INITIAL[event[0]] if event[0] in self.params.AC_WIP_INITIAL else 0
            ac_wip = initial_ac_wip + 0#resource_task.count

            duration = self.process.get_predict_processing(str(self.id), pr_wip, transition, ac_wip, rp_oc, self.start_time + timedelta(seconds=env.now), -1)
            if self.NAME_EXPERIMENT == 'Productions':
                attrib[-1] = duration/60
            attrib.append(event[-2])  # trace attributes
            attrib.append(event[-1])  # label
            if event[1] == -1:
                yield env.timeout(duration)
            else:
                if event[1] < 0:
                    event[1] = 0
                yield env.timeout(event[1])
            buffer.append(str(self.start_time + timedelta(seconds=env.now))[:19])
            buffer.append(resource.get_name())
            buffer.append(pr_wip_wait)
            buffer.append(ac_wip)
            buffer.append(queue)
            buffer = buffer + attrib
            resource_task.release(resource_task_request)
            print(*buffer)
            writer.writerow(buffer)
            resource.release(request_resource)
            #self.update_marking(trans)
            #trans = self.next_transition(syn)
            self.see_activity = True
            event = self.next_event() if self.contrafactual == False else self.next_event_contrafactual()

        resource_trace.release(resource_trace_request)


    def next_event_contrafactual(self):
        if self.contrafactual:
            if self.sequence and self.sequence[0][0] == self.contrafactual[0][0]:
                next = self.sequence[0]
            else:
                # [event, event_processingTime, resource, wait, attrib_events, attrib_traces, label]
                next = [self.contrafactual[0][0], -1, self.contrafactual[0][1], -1, self.contrafactual[0][-3], self.contrafactual[0][-2], self.contrafactual[0][-1]]
            if self.sequence:
                del self.sequence[0]
            del self.contrafactual[0]
        else:
            next = None
        return next

    def next_event(self):
        if self.sequence:
            next = self.sequence[0]
            del self.sequence[0]
        else:
            next = None
        return next

    def update_marking(self, trans):
        self.am = semantics.execute(trans, self.net, self.am)

    def next_transition(self, syn):
        all_enabled_trans = semantics.enabled_transitions(self.net, self.am)
        all_enabled_trans = list(all_enabled_trans)
        all_enabled_trans.sort(key=lambda x: x.name)
        label_element = str(list(self.am)[0])
        if len(all_enabled_trans) == 0:
            return None
        elif len(all_enabled_trans) > 1:
            if syn:
                prob = [0.50, 0.50]
                if label_element in ['exi_node_54ded9af-1e77-4081-8659-bd5554ae9b9d', 'exi_node_38c10378-0c54-4b13-8c4c-db3e4d952451', 'exi_node_52e223db-6ebf-4cc7-920e-9737fe97b655', 'exi_node_e55f8171-c3fc-4120-9ab1-167a472007b7']:
                    prob = [0.01, 0.99]
                elif label_element in ['exi_node_7af111c0-b232-4481-8fce-8d8278b1cb5a']:
                    prob = [0.99, 0.01]
                value = [*range(0, len(prob), 1)]
                next = int(random.choices(value, prob)[0])
            else:
                next = random.choices(list(range(0, len(all_enabled_trans), 1)))[0]
            return all_enabled_trans[next]
        else:
            return all_enabled_trans[0]

