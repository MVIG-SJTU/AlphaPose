-- Designed only for loop-free models
local MRF, parent = torch.class('MRF', 'nn.Module')

function gen_pairwise_ref(conn)
    local pw_ref = {}
    local inv_ref = {}
    local num_nodes = conn:size()[1]
    local count = 1
    for i = 1,num_nodes do
        for j = i+1,num_nodes do
            if conn[i][j] == 1 then
                if pw_ref[i] == nil then pw_ref[i] = {} end
                if pw_ref[j] == nil then pw_ref[j] = {} end
                pw_ref[i][j] = count
                pw_ref[j][i] = -count
                inv_ref[count] = {i,j}
                count = count + 1
            end
        end
    end
    return pw_ref,inv_ref
end

function MRF:__init(connections, num_states)
    -- No error checking here, but connections should be 0 along the diagonal
    -- and connections should equal connections:transpose
    parent.__init(self)
    self.num_nodes = connections:size()[1]
    self.num_states = num_states
    self.num_edges = connections:sum() / 2
    self.conn = torch.Tensor(connections:size()):copy(connections)
    self.all_idxs = torch.linspace(1,self.num_nodes,self.num_nodes)
    self.pw_ref,_ = gen_pairwise_ref(connections)
    self.do_full_reset = true
end

function MRF:reset(input, batch_size)
    if self.do_full_reset then
        -- Allocate memory for everything once
        self.unary = torch.Tensor(batch_size, self.num_nodes, self.num_states)
        self.pairwise = torch.Tensor(batch_size, self.num_edges, self.num_states, self.num_states)
        self.msg = torch.Tensor(batch_size, self.num_nodes, self.num_nodes, 2, self.num_states)
        self.msg_flag = torch.zeros(batch_size, self.num_nodes, self.num_nodes)
        self.z = torch.Tensor(batch_size, self.num_nodes)
        self.temp_output = torch.Tensor(self.batch_size, self.num_nodes * self.num_states)
        self.temp_gradInput = torch.zeros(self.batch_size, input[1]:numel())
        -- In case everything is else is on GPU
        self.output = torch.Tensor(self.batch_size, self.num_nodes * self.num_states):typeAs(input)
        self.gradInput = torch.zeros(self.batch_size, input[1]:numel()):typeAs(input)
        if self.out_vector then
            -- Extra stuff to handle receiving a single vector as input properly
            self.output = self.output:view(out:size(2))
            self.gradInput = self.gradInput:view(self.gradInput:numel())
        end
    else
        self.msg_flag:fill(0)
    end
end

function MRF:set_potentials(unary, pairwise)
    self.unary:copy(unary)
    self.pairwise:copy(pairwise)
end

function MRF:get_neighbors(node_idx, exclude)
    local idxs = self.all_idxs[self.conn[node_idx]:eq(1)]
    if exclude ~= nil then idxs = idxs[idxs:ne(exclude)] end
    return idxs
end

function MRF:get_message(sample_idx, node_idx, exclude)
    local nb = self:get_neighbors(node_idx, exclude)
    local msg = self.msg[sample_idx][node_idx]
    local msg_flag = self.msg_flag[sample_idx][node_idx]
    if nb:nDimension() == 0 then
        -- Leaf node, return unary scores
        return self.unary[sample_idx][node_idx]
    else
        -- Get message from all neighbors
        local out_msg = torch.ones(self.num_states):typeAs(self.unary)
        for i = 1,nb:numel() do
            if msg_flag[nb[i]] == 0 then
                -- Get pairwise scores
                local pw_idx = self.pw_ref[node_idx][nb[i]]
                local pw_vals = self.pairwise[sample_idx][math.abs(pw_idx)]
                if pw_idx < 0 then pw_vals = pw_vals:transpose(1,2) end
                -- Get message from neighbor
                msg[nb[i]][1] = self:get_message(sample_idx, nb[i], node_idx)
                msg[nb[i]][2] = pw_vals * msg[nb[i]][1]
                msg_flag[nb[i]] = 1
            end
            out_msg:cmul(msg[nb[i]][2])
        end
    	out_msg:cmul(self.unary[sample_idx][node_idx])
        if exclude == nil then
            -- The only reason to keep z per node is in case a graph is provided that isn't connected
            -- It doesn't add much overhead, so it is easy enough to keep, instead of restricting to
            -- only connected graphs
            self.z[sample_idx][node_idx] = out_msg:sum()
            out_msg:div(self.z[sample_idx][node_idx])
        end
        return out_msg
    end
end

function MRF:updateOutput(input)
    local num_in = self.num_states*(self.num_nodes + self.num_edges * self.num_states)
    -- If only presented with a 1D vector, change the view so it is 2D
    if input:nDimension() == 1 then
        input = input:view(1,input:size()[1])
        self.out_vector = true
    else
        self.out_vector = false
    end
    if input:nDimension() > 2 then
        -- No more than two dimensions expected
        print("Bad number of input dimensions to graphical model (must be 1 or 2)")
    elseif input[1]:numel() ~= num_in then
        -- Input should be a vector of size unary:numel() + pairwise:numel()
        print("Input has wrong number of elements")
        print(num_in .. " expected.")
        print(input[1]:numel() .. " received.")
    else
        if self.batch_size ~= input:size(1) then
            -- If we haven't done it before, or batch size changes
            -- Initialize batch size and output tensor dimensions
            self.batch_size = input:size(1)
            self.do_full_reset = true
        end

        -- Reset, important for message flags
        self:reset(input, self.batch_size)
        -- Set potentials
        local un = input:sub(1,-1,1,self.unary[1]:numel())
        local pw = input:sub(1,-1,self.unary[1]:numel()+1,-1)
        self:set_potentials(un,pw)
        -- For each sample in the batch calculate output
        for i = 1,self.batch_size do
            for j = 1,self.num_nodes do
                local start_idx = (j-1)*self.num_states + 1
                self.temp_output[i][{{start_idx,start_idx+self.num_states-1}}] = self:get_message(i,j)
            end
        end

        if (torch.eq(self.temp_output,self.temp_output):sum() ~= self.temp_output:numel()) then
            print(input)
            print(self.temp_output)
            print("nan present in output")
            assert(false)
        end

        -- Copy output to GPU
        self.output:copy(self.temp_output)
        return self.output
    end
end

function MRF:gradient_message(sample_idx, node_idx, curr_grad, exclude)
    -- Change how we see the gradient input vector to ease data access
    local out = self.temp_output[sample_idx]:view(self.unary[1]:size())
    local grad_in_un = self.temp_gradInput[sample_idx]:sub(1,self.unary[1]:numel()):view(self.unary[1]:size())
    local grad_in_pw = self.temp_gradInput[sample_idx]:sub(self.unary[1]:numel()+1,-1):view(self.pairwise[1]:size())
    -- Get node neighbors and messages from forward pass
    local nb = self:get_neighbors(node_idx, exclude)
    local in_msg = self.msg[sample_idx][node_idx]
    local out_msg
    if exclude == nil then out_msg = out[node_idx]*self.z[sample_idx][node_idx]
    else out_msg = self.msg[sample_idx][exclude][node_idx][1] end

    if nb:nDimension() == 0 then
        -- Leaf node, no neighbors, update all the unary score gradients
        if (torch.eq(curr_grad,curr_grad):sum() ~= self.num_states) then
            print("There was a nan in the passed gradient....")
            print(curr_grad)
        end
        grad_in_un[node_idx]:add(curr_grad)
    else
        -- Else, pass gradient message back to each neighbor before updating unary scores
        local temp_grad = torch.cmul(curr_grad, out_msg)
        for i = 1,nb:size()[1] do
            local temp_grad_2 = torch.cdiv(temp_grad, in_msg[nb[i]][2])
            if (torch.eq(temp_grad_2,temp_grad_2):sum() ~= self.num_states) then
                print("Temp_grad_2 nan...")
                print(curr_grad)
            end
            local pw_idx = self.pw_ref[node_idx][nb[i]]
            local pw = self.pairwise[sample_idx][math.abs(pw_idx)]
            if pw_idx > 0 then pw = pw:transpose(1,2) end
            -- Pass message back
            self:gradient_message(sample_idx, nb[i], pw * temp_grad_2, node_idx)
            -- Calculate pairwise gradient
            local pw_grad = torch.ger(temp_grad_2, in_msg[nb[i]][1]) -- Vector outer product
            if node_idx > nb[i] then pw_grad = pw_grad:transpose(1,2) end
            grad_in_pw[math.abs(pw_idx)]:add(pw_grad)
        end
        temp_grad:cdiv(self.unary[sample_idx][node_idx])
        temp_grad[torch.eq(self.unary[sample_idx][node_idx],0)] = 0
        if (torch.eq(temp_grad,temp_grad):sum() ~= self.num_states) then
            print("There was a nan in temp gradient...still?")
            print(curr_grad)
        end
        grad_in_un[node_idx]:add(temp_grad)
    end
end

function MRF:updateGradInput(input, gradOutput)
    -- If only presented with a 1D vector, change the view so it is 2D
    if input:nDimension() == 1 then
        gradOutput = gradOutput:view(1, gradOutput:numel())
    end

    for i = 1,self.batch_size do
        -- For each sample, loop through each node
        -- The outer loop is embarrassingly parallel, the inner loop can easily be as well (at the moment it is not)
        local out = self.temp_output[i]:view(self.num_nodes,self.num_states)
        local grad_out = gradOutput[i]:view(self.num_nodes,self.num_states):double()
        for j = 1,self.num_nodes do
            -- To start handle gradient calculation taking into account normalization by z
            local temp_mat = -torch.repeatTensor(out[j],self.num_states,1) + torch.diag(torch.ones(self.num_states))
            local temp_grad = (temp_mat * grad_out[j]) / self.z[i][j]
            -- Then send gradient message back through nodes
            self:gradient_message(i, j, temp_grad, nil)
        end
    end

    if (torch.eq(self.temp_gradInput,self.temp_gradInput):sum() ~= self.temp_gradInput:numel()) then
        print(self.temp_output)
        print(gradOutput)
        print(self.temp_gradInput)
        print("nan present")
        assert(false)
    end

    self.gradInput:copy(self.temp_gradInput)
    return self.gradInput
end