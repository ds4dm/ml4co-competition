# generate parameter files
def writeParameterFile(paramfile):
    file = open(paramfile,'r').readlines()

    parameters = ''
    nextparam = True
    i = 1
    while i < len(file) - 1:
        
        # check if the given file has the right format
        if not file[i].startswith('# [type:'):
            print(file[i])
            print("file has not the right format!")
            return

        paramtype = file[i].split(' ')[2][:-1]
        
        # 'string'-settings are not for tuning
        if paramtype == 'string':
            i += 4
            continue

        paramrange = file[i].split(' ')[6][:-1]
        default = '[' + file[i].split(' ')[8][:-1]
        name = file[i + 1].split(' ')[0]

        # if parameter has type 'char', we need to transform its range into right format for SMAC
        if paramtype == 'char':
            elements = paramrange[1:-1]
            paramrange = '{'
            
            for j in range(len(elements)):
                paramrange += elements[j] + ','
            
            paramrange = paramrange[:-1]
            paramrange += '}' 

        # change 'char' to 'catigorical'
        if paramtype == 'bool' or paramtype == 'char':
            paramtype = 'categorical'
        elif paramtype == 'int':
            paramtype = 'integer'

        parameters += name + ' ' + paramtype + ' ' + paramrange + ' ' + default + '\n'

        # move on to next parameter
        i += 4

    # add settings to 'parameters.txt'
    paramfile = open('parameters.pcs', "a+")
    paramfile.write(parameters)
    paramfile.close()

    return


writeParameterFile('parameters_to_tune.txt')
